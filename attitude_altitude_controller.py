# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/quadcopter.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort:skip

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, device):
        self.kp = kp
        self.ki = ki 
        self.kd = kd
        self.integral = torch.tensor(0.0, device=device)
        self.prev_error = torch.tensor(0.0, device=device)
        
    def update(self, error: torch.Tensor, dt: float) -> torch.Tensor:
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

class QuadcopterController:
    def __init__(self, robot_mass: float, J: torch.Tensor, device, gravity=9.81):
        self.mass = robot_mass
        self.J = J
        self.g = gravity
        self.device = device
        
        # PID controllers for roll, pitch, yaw
        self.roll_pid = PIDController(kp=0.4, ki=0.07, kd=0.1, device=device)
        self.pitch_pid = PIDController(kp=0.4, ki=0.05, kd=0.1, device=device)
        self.yaw_pid = PIDController(kp=1.2, ki=0.05, kd=0.1, device=device)
        
        # PID controller for altitude
        self.alt_pid = PIDController(kp=5.0, ki=0.05, kd=1.0, device=device)

    def control(self, 
                desired_attitude: torch.Tensor,
                desired_altitude: torch.Tensor,
                current_attitude: torch.Tensor,
                current_altitude: torch.Tensor, 
                R_robot: torch.Tensor,
                dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Calculate attitude errors
        attitude_error = desired_attitude - current_attitude
        
        # Get PID outputs for attitude
        moment_scale = 0.007
        roll_moment = self.roll_pid.update(attitude_error[0], dt)
        pitch_moment = self.pitch_pid.update(attitude_error[1], dt)
        yaw_moment = self.yaw_pid.update(attitude_error[2], dt)
        moment = torch.tensor([roll_moment, pitch_moment, yaw_moment], device=self.device)*moment_scale
        
        # Calculate required thrust considering attitude
        z_world = R_robot @ torch.tensor([0.0, 0.0, 1.0], device=self.device)
        altitude_error = desired_altitude - current_altitude
        
        # PID output for altitude
        altitude_force = self.alt_pid.update(altitude_error, dt)
        
        # Total thrust needed
        total_thrust = (self.mass * self.g + altitude_force) / torch.dot(z_world, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        
        return total_thrust, moment

def QuaternionToRotationMatrix(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    x2, y2, z2 = x * 2, y * 2, z * 2
    wx, wy, wz = w * x2, w * y2, w * z2
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    return torch.tensor([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)]
    ], device=quat.device)

def QuaternionToRPY(quat: torch.Tensor) -> torch.Tensor:
    # Unit: radian
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    if w == 0.0:
        return torch.tensor([0.0, 0.0, 0.0], device=quat.device)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    t2 = 2.0 * (w * y - z * x)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    if t0 == 0.0 and t1 == 0.0:
        roll = 0.0
    else:
        roll = torch.atan2(t0, t1)
    pitch = torch.asin(t2)
    if t3 == 0.0 and t4 == 0.0:
        yaw = 0.0
    else:
        yaw = torch.atan2(t3, t4)

    return torch.tensor([roll, pitch, yaw], device=quat.device)

def RpyToQuaternion(rpy: torch.Tensor) -> torch.Tensor:
    # Unit: radian
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    return torch.tensor([
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
        cy * cr * cp + sy * sr * sp
    ], device=rpy.device)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)

    # create handles for the robots
    robot = Articulation(robot_cfg)
    # Play the simulator
    sim.reset()

    # Fetch relevant parameters to make the quadcopter hover in place
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    robot_rotational_inertia = robot.root_physx_view.get_inertias()
    print(f"Robot rotational inertia: {robot_rotational_inertia}")
    print(f"Robot mass: {robot_mass}")
    gravity :float = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    # joint position with respect to the mass center :
    # ( 0.031,-0.031，0.021)
    # (-0.031,-0.031，0.021)
    # (-0.031, 0.031，0.021)
    # ( 0.031, 0.031，0.021)

    # 等效转动惯量
    #([[2.1066e-05, 0.0000e+00, 0.0000e+00],
    #  [0.0000e+00, 2.1810e-05, 0.0000e+00],
    #  [0.0000e+00, 0.0000e+00, 3.6084e-05]])
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    robot_J = torch.tensor([
        [2.1066e-05, 0.0, 0.0],
        [0.0, 2.1810e-05, 0.0],
        [0.0, 0.0, 3.6084e-05]
    ], device=sim.device)
    
    # Initialize controller
    controller = QuadcopterController(robot_mass=robot_mass, J=robot_J, device=sim.device, gravity=gravity)

    # Define desired states
    desired_altitude = torch.tensor(2.0, device=sim.device)  # Hover at 1m
    desired_attitude = torch.zeros(3, device=sim.device)     # Level flight

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 800 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            root_pos = robot.data.default_root_state[:, :7]
            # generate random rotate on reset
            root_rpy = torch.rand(3, device=sim.device) *3.1415926/4.0
            root_quat = RpyToQuaternion(root_rpy)
            root_pos[:, 3:] = root_quat
            robot.write_root_pose_to_sim(root_pos)
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            
            # reset command
            print(">>>>>>>> Reset!")
        # Get current state
        current_pos = robot.data.body_pos_w[0][0]
        current_quat = robot.data.body_quat_w[0][0]
        R_robot = QuaternionToRotationMatrix(current_quat)
        
        # Extract current attitude (you'll need to implement this based on quaternion)
        current_attitude = QuaternionToRPY(current_quat)
        current_vel = robot.data.body_lin_vel_w[0][0]
        print(f"Current position: {current_pos}, current attitude: {current_attitude}, current velocity: {current_vel}")
        current_altitude = current_pos[2]
        
        # Get control outputs
        total_thrust, moments = controller.control(
            desired_attitude, desired_altitude,
            current_attitude, current_altitude,
            R_robot, sim_dt
        )

        # Apply controls
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)
        total_thrust = total_thrust.clamp(-robot_mass*gravity*0.5,robot_mass*gravity*1.5)
        moments[...,2]*=2
        moments = moments.clamp(-0.02,0.02)
        forces[..., 2] = total_thrust / 4.0
        torques[..., 0] = moments[0]
        torques[..., 1] = moments[1]
        torques[..., 2] = moments[2]
        
        # Special handling for z-axis torque
        forces[..., 0, 0] =  moments[2] / 0.031 / 2.0
        forces[..., 2, 0] = -moments[2] / 0.031 / 2.0
        torques[..., 2] = 0.0

        # write the command to the robot
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
