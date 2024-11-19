"""Launch the Simulator first."""
import argparse
import torch
import numpy as np
import random

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tune PID parameters for velocity and attitude controller using a vectorized approach.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort:skip

@configclass
class UAVSceneCfg(InteractiveSceneCfg):
    """Configuration for a UAV scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Crazyflie")

class Individual:
    def __init__(self):
        self.vel_pid_range = {
            "K_p_max": torch.tensor([0.15, 0.15, 0.15], device=args_cli.device),
            "K_i_max": torch.tensor([0.01, 0.01, 0.01], device=args_cli.device),
            "K_d_max": torch.tensor([0.05, 0.05, 0.05], device=args_cli.device),
        }
        self.att_pd_range = {
            "K_p_max": torch.tensor([0.1, 0.1, 0.1], device=args_cli.device),
            "K_d_max": torch.tensor([0.01, 0.01, 0.01], device=args_cli.device),
        }
        # Randomize parameters 
        self.vel_pid = {
            "K_p": torch.rand(3, device=args_cli.device) * self.vel_pid_range["K_p_max"],
            "K_i": torch.rand(3, device=args_cli.device) * self.vel_pid_range["K_i_max"],
            "K_d": torch.rand(3, device=args_cli.device) * self.vel_pid_range["K_d_max"],
        }
        self.att_pd = {
            "K_p": torch.rand(3, device=args_cli.device) * self.att_pd_range["K_p_max"],
            "K_d": torch.rand(3, device=args_cli.device) * self.att_pd_range["K_d_max"],
        }
        self.fitness = 0.0
        self.integral_error = torch.zeros(3, device=args_cli.device)
    
    def update_fitness(self, fitness):
        self.fitness = fitness

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, parents):
    """Runs the simulation loop."""
    # Extract scene entities
    robot: Articulation = scene["robot"]

    # Fetch relevant parameters 
    sim_dt = sim.get_physics_dt()
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()/robot.num_instances
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    # buffer for forces and torques
    forces = torch.zeros(robot.num_instances, 4, 3, device=args_cli.device)
    torques = torch.zeros_like(forces)

    count = 1
    offspring = parents
    desired_velocity = torch.zeros(args_cli.num_envs, 3, device=args_cli.device)
    velocity_limits = torch.tensor([5.0, 5.0, 5.0], device=args_cli.device)

    while simulation_app.is_running():
        sim_time = count * sim_dt
        # reset the env every 1000 steps
        if count % 500 == 0:
            if count % 5000 == 0:
                # save the best individual to a a file in a readable format
                offspring.sort(key=lambda ind: ind.fitness, reverse=True)
                best_individual = offspring[0]
                with open("best_individual.txt", "w") as f:
                    f.write(f"Velocity PID parameters: {best_individual.vel_pid}")
                    f.write(f"Attitude PD parameters: {best_individual.att_pd}")
                    f.write(f"Fitness: {best_individual.fitness}")
            print(f"Sim time: {sim_time:.2f}")
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_state_to_sim(root_state)
            scene.reset()
            print(">>>>>>>> Reset!")
            # evaluate fitness for the current generation
            for i, individual in enumerate(offspring):
                # compute fitness based on accumulated error
                individual.update_fitness(-individual.integral_error.norm().item())
                # reset integral error
                individual.integral_error = torch.zeros(3, device=args_cli.device)
            # print out some indicators
            print("Generation:", count // 500)
            print("Best fitness:", offspring[0].fitness)
            print("Average fitness:", np.mean([ind.fitness for ind in offspring]))
            # get the next generation of PID parameters
            offspring = next_gen(offspring)
            # reset the desired velocity each evaluation
            desired_velocity = torch.rand(args_cli.num_envs, 3, device=args_cli.device) * velocity_limits
            # print("Desired velocity:", desired_velocity)
        # calculate the fitness of the current generation every step based on errors
        # get current velocities
        current_velocity = robot.data.root_lin_vel_w  # shape: (num_envs, 3)
        # compute errorgravi
        velocity_error = desired_velocity - current_velocity  # shape: (num_envs, 3)
        # accumulate integral error for fitness evaluation
        for i, individual in enumerate(offspring):
            individual.integral_error += velocity_error[i] * sim_dt
        # calculate the forces and torques through PID controllers
        forces, torques = get_command(offspring, velocity_error, robot,sim)
        # apply the forces and torques to the robot
        robot.set_external_force_and_torque(forces, torques, body_ids=prop_body_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

def next_gen(parents: list):
    """
    Get the next generation of PID parameters.
    """
    offspring = []
    elite_rate = 0.1
    mutation_rate = 0.2
    num_elites = int(len(parents) * elite_rate)
    # sort parents by fitness
    parents.sort(key=lambda ind: ind.fitness, reverse=True)
    elites = parents[:num_elites]
    # generate offspring
    while len(offspring) < len(parents):
        # selection (tournament)
        parent1 = tournament_selection(parents)
        parent2 = tournament_selection(parents)
        # crossover
        child = crossover(parent1, parent2)
        # mutation
        mutate(child, mutation_rate)
        offspring.append(child)
    # replace worst individuals with elites
    offspring[-num_elites:] = elites
    return offspring

def tournament_selection(population):
    """Select an individual by tournament selection."""
    k = 3
    selected = random.sample(population, k)
    selected.sort(key=lambda ind: ind.fitness, reverse=True)
    return selected[0]

def crossover(parent1, parent2):
    """Crossover two parents to create a child."""
    child = Individual()
    for param in ['K_p', 'K_i', 'K_d']:
        # Velocity PID parameters
        child.vel_pid[param] = (parent1.vel_pid[param] + parent2.vel_pid[param]) / 2
    for param in ['K_p', 'K_d']:
        # Attitude PD parameters
        child.att_pd[param] = (parent1.att_pd[param] + parent2.att_pd[param]) / 2
    return child

def mutate(individual, mutation_rate):
    """Mutate an individual's PID parameters."""
    if random.random() < mutation_rate:
        for param in ['K_p', 'K_i', 'K_d']:
            individual.vel_pid[param] += torch.randn(3, device=args_cli.device) * 0.2
            # Ensure parameters stay within the defined range
            individual.vel_pid[param] = torch.clamp(
                individual.vel_pid[param], max=individual.vel_pid_range[param + "_max"]
            )
        for param in ['K_p', 'K_d']:
            individual.att_pd[param] += torch.randn(3, device=args_cli.device) * 0.2
            individual.att_pd[param] = torch.clamp(
                individual.att_pd[param], max=individual.att_pd_range[param + "_max"]
            )

def get_command(population, velocity_error, robot: Articulation, sim: sim_utils.SimulationContext):
    sim_dt = sim.get_physics_dt()
    num_envs = args_cli.num_envs
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    robot_mass = robot.root_physx_view.get_masses().sum() / robot.num_instances

    # Set limits
    force_limit = gravity * robot_mass / 4.0 * 2
    torque_limit = 0.01

    # Consolidate parameters into tensors (num_envs, 3)
    vel_Kp = torch.stack([ind.vel_pid["K_p"] for ind in population])
    vel_Ki = torch.stack([ind.vel_pid["K_i"] for ind in population])
    vel_Kd = torch.stack([ind.vel_pid["K_d"] for ind in population])
    att_Kp = torch.stack([ind.att_pd["K_p"] for ind in population])
    att_Kd = torch.stack([ind.att_pd["K_d"] for ind in population])

    # Update and clamp integral errors (num_envs, 3)
    for i, ind in enumerate(population):
        # Use individual velocity error for each environment
        ind.integral_error += velocity_error[i] * sim_dt  
        ind.integral_error = torch.clamp(ind.integral_error, -5.0, 5.0)
    integral_errors = torch.stack([ind.integral_error for ind in population])
    
    # Velocity PID control (vectorized)
    derivative_error = -robot.data.root_lin_vel_w
    pid_output = (
        vel_Kp * velocity_error +
        vel_Ki * integral_errors +
        vel_Kd * derivative_error
    )
    pid_output = torch.clamp(pid_output, -10.0, 10.0)

    # Compute desired acceleration and thrust
    gravity_compensation = torch.zeros(num_envs, 3, device=args_cli.device)
    gravity_compensation[:, 2] = gravity/4.0
    desired_acceleration = pid_output + gravity_compensation
    total_thrust = robot_mass * desired_acceleration
    
    # Reshape forces for 4 propellers per drone
    forces = torch.clamp(total_thrust, -force_limit, force_limit)
    forces = forces.unsqueeze(1).repeat(1, 4, 1)  # Shape: (num_envs, 4, 3)

    # Attitude PD control (vectorized)
    current_orientation = robot.data.root_quat_w
    desired_orientation = torch.tile(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=args_cli.device),
        (num_envs, 1)
    )
    orientation_errors = torch.stack([quat_error(desired_orientation[i], current_orientation[i]) 
                                   for i in range(num_envs)])
    angular_velocity = robot.data.root_ang_vel_w

    pd_output = (
        att_Kp * orientation_errors +
        att_Kd * (-angular_velocity)
    )
    torques = torch.clamp(pd_output, -torque_limit, torque_limit)
    torques = torques.unsqueeze(1).repeat(1, 4, 1)  # Shape: (num_envs, 4, 3)

    return forces, torques

def quat_error(qd, q):
    """Compute quaternion error."""
    qe = torch.zeros(3, device=args_cli.device)
    qe[0] = qd[0]*q[1] - qd[1]*q[0] - qd[2]*q[3] + qd[3]*q[2]
    qe[1] = qd[0]*q[2] + qd[1]*q[3] - qd[2]*q[0] - qd[3]*q[1]
    qe[2] = qd[0]*q[3] - qd[1]*q[2] + qd[2]*q[1] - qd[3]*q[0]
    return qe

def main():
    """Main function."""
    # Initialize the Population
    num_individuals = args_cli.num_envs
    parents = [Individual() for _ in range(num_individuals)]

    sim_cfg = sim_utils.SimulationCfg(dt=0.003, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])
    # Create scene
    scene_cfg = UAVSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    run_simulator(sim, scene, parents)

if __name__ == "__main__":
    # Run the main function
    main()
    simulation_app.close()