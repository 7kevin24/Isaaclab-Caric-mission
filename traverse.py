"""
This Env defines a UAVs Traversal environment where multiple UAVs are tasked to traverse a cubic area.
"""

from __future__ import annotations
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg
from omni.isaac.lab.envs.common import AgentID

from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms

from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

import torch
from .controller import AttAltController, PositionController, VelocityController, QuaternionToRPY, QuaternionToRotationMatrix

# Import controllers from controller.py

@configclass
class UAVsTraversalEnvCfg(DirectMARLEnvCfg):
    # Environment settings
    episode_length_s = 20.0
    decimation = 5  # Inner loop frequency is 5 times the outer loop

    # Area bounds
    area_bounds = {
        "min": [-2.0, -2.0, 1],
        "max": [2.0, 2.0, 5.0]
    }
    area_size = (
        (area_bounds["max"][0] - area_bounds["min"][0]),
        (area_bounds["max"][1] - area_bounds["min"][1]),
        (area_bounds["max"][2] - area_bounds["min"][2]),
    )

    # Grid map settings
    grid_resolution = 1.0  # size of each grid cell
    grid_size = (
        int(area_size[0] / grid_resolution),
        int(area_size[1] / grid_resolution),
        int(area_size[2] / grid_resolution),
    )

    # Agent settings
    num_agents: int = 2
    possible_agents = [f"agent_{i}" for i in range(num_agents)]

    # Action Space: [x,y,z] shape:3
    action_spaces = {agent: 3 for agent in possible_agents}
    # Observation Space: [position_w, other_agents_pos, grid_map_flat]
    obs_shape = 3 * num_agents + grid_size[0]*grid_size[1]*grid_size[2]
    observation_spaces = {agent: 70 for agent in possible_agents}

    state_space = obs_shape

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=5.0,
        replicate_physics=True,
    )

    # Controller settings
    thrust_to_weight = 1.5
    moment_scale = 0.01

    # Reward scales
    distance_reward_scale = 1.0
    collision_penalty = -10.0
    visit_reward_scale = 1.0  # Added reward scale for visiting new grid cells

    # Robot configuration
    robot = {
        agent: CRAZYFLIE_CFG.replace(
            prim_path=f"/World/envs/env_.*/Robot_{agent}"
        ).replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0 * (count + 1)),
                rot=(1.0, 0.0, 0.0, 0.0),
            )
        )
        for agent, count in zip(possible_agents, range(num_agents))
    }


class UAVsTraversalEnv(DirectMARLEnv):
    cfg: UAVsTraversalEnvCfg

    def __init__(self, cfg: UAVsTraversalEnvCfg, **kwargs):
        self.possible_agents = cfg.possible_agents
        super().__init__(cfg, **kwargs)

        # Define the cubic area using two vertices
        self.area_bounds = {
            "min": torch.tensor(cfg.area_bounds["min"], device=self.device),
            "max": torch.tensor(cfg.area_bounds["max"], device=self.device),
        }
        self.area_size = torch.tensor(cfg.area_size, device=self.device)
        self.grid_resolution = cfg.grid_resolution
        self.grid_size = cfg.grid_size

        # Initialize grid maps for each environment
        self.grid_map = torch.zeros(
            self.num_envs,
            self.grid_size[0],
            self.grid_size[1],
            self.grid_size[2],
            device=self.device,
        )

        # Initialize controllers
        self.desired_attitude = torch.zeros(self.num_envs, 3, device=self.device)
        self.desired_altitude = torch.ones(self.num_envs, device=self.device) * 1.0
        self.controllers = {}
        self.dt = self.sim.get_physics_dt()
        for agent in self.possible_agents:
            self.controllers[agent] = {
                "position": PositionController(device=self.device),
                "velocity": VelocityController(device=self.device),
                "attitude": AttAltController(
                    robot_mass=self._robots[agent].root_physx_view.get_masses()[0].sum(),
                    J=torch.tensor(
                        [
                            [2.1066e-05, 0.0, 0.0],
                            [0.0, 2.1810e-05, 0.0],
                            [0.0, 0.0, 3.6084e-05],
                        ],
                        device=self.device,
                    ),
                    device=self.device,
                ),
            }

        # Initialize action tensors
        self._actions = {
            agent: torch.zeros(self.num_envs, self.cfg.action_spaces[agent], device=self.device)
            for agent in self.possible_agents
        }

        # Robots and body indices
        self._robots = {}
        self._body_ids = {}
        self._robot_masses = {}
        self._robot_weights = {}
        for agent in self.possible_agents:
            self._robots[agent] = self.scene.articulations[str(agent)]
            self._body_ids[agent] = self._robots[agent].find_bodies("body")[0]
            self._robot_masses[agent] = self._robots[agent].root_physx_view.get_masses()[0].sum()
            gravity_norm = torch.norm(torch.tensor(self.sim.cfg.gravity))
            self._robot_weights[agent] = (self._robot_masses[agent] * gravity_norm).item()

    def _setup_scene(self):
        # Create terrain first
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Initialize robots dictionary
        self._robots: dict[AgentID, Articulation] = {}

        # Create robots after terrain
        for agent, robot_cfg in self.cfg.robot.items():
            # Create the robot
            self._robots[agent] = Articulation(robot_cfg)
            # Register with scene manager
            self.scene.articulations[agent] = self._robots[agent]

        # Clone environments after all assets are created
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reset_idx(self, env_ids: torch.Tensor):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        for agent in self.possible_agents:
            # Reset robot states
            default_state = self._robots[agent].data.default_root_state.clone()[env_ids]
            default_state[:, :3] += self.scene.env_origins[env_ids]
            self._robots[agent].write_root_state_to_sim(default_state, env_ids)

            # Reset controllers
            self.controllers[agent]["position"].clear_integral()
            self.controllers[agent]["velocity"].clear_integral()
            self.controllers[agent]["attitude"].clear_integral()

            # Set new waypoints within the area
            self._actions[agent][env_ids] = torch.rand(len(env_ids), 3, device=self.device) * (
                self.area_bounds["max"] - self.area_bounds["min"]
            ) + self.area_bounds["min"]

        # Reset grid map
        self.grid_map[env_ids] = 0

        # Reset buffers
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        super()._reset_idx(env_ids)

    def _pre_physics_step(self, actions):
        sim_dt = self.sim.get_physics_dt()

        # Outer loop: Position and Velocity Control
        for agent in self.possible_agents:
            current_pos = self._robots[agent].data.root_pos_w - self.scene.env_origins
            # print("current_pos", current_pos[0], current_pos[1], current_pos[2])
            current_vel = self._robots[agent].data.root_lin_vel_w
            desired_pos = actions[agent][:, :3]
            # generate random actions to test
            desired_pos = torch.rand(len(desired_pos), 3, device=self.device) * (
                self.area_bounds["max"] - self.area_bounds["min"]
            ) + self.area_bounds["min"]
            desired_pos = torch.clamp(desired_pos, self.area_bounds["min"], self.area_bounds["max"])
            current_quat = self._robots[agent].data.root_quat_w
            # Position controller to compute desired velocity
            desired_vel = self.controllers[agent]["position"].control(
                desired_pos, current_pos,current_quat ,self.dt
            )

            # Velocity controller to compute desired attitude
            self.desired_attitude = self.controllers[agent]["velocity"].control(
                desired_vel, current_vel,self.dt
            )
            self.desired_altitude = desired_pos[:, 2]

    def _apply_action(self):
        # Inner loop: Attitude and Altitude Control
        for agent in self.possible_agents:
            current_quat = self._robots[agent].data.root_quat_w
            current_attitude = QuaternionToRPY(current_quat)
            current_altitude = self._robots[agent].data.root_pos_w[:, 2]
            R_robot = QuaternionToRotationMatrix(current_quat)

            total_thrust, moments = self.controllers[agent]["attitude"].control(
                desired_attitude=self.desired_attitude,
                desired_altitude=self.desired_altitude,
                current_attitude=current_attitude,
                current_altitude=current_altitude,
                R_robot=R_robot,
                dt=self.dt,
            )

            # Apply forces and torques
            forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
            torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
            forces[:, 0, 2] = total_thrust
            torques[:, 0, :] = moments
            self._robots[agent].set_external_force_and_torque(
                forces, torques, body_ids=self._body_ids[agent]
            )

    def _get_rewards(self):
        rewards = {}
        for agent in self.possible_agents:
            current_pos = self._robots[agent].data.root_pos_w - self.scene.env_origins
            desired_pos = self._actions[agent][:, :3]
            distance = torch.norm(desired_pos - current_pos, dim=-1)
            progress_reward = -distance * self.cfg.distance_reward_scale

            # Collision penalty (if any) TODO: collision detection
            # collision = self._robots[agent].data.contact_forces.norm(dim=-1) > 0.0
            # collision_penalty = collision.float() * self.cfg.collision_penalty
            collision_penalty = 0.0 
            # Compute whether agent has visited a new grid cell
            grid_idx = ((current_pos - self.area_bounds["min"]) / self.grid_resolution).long()
            max_grid_idx = torch.tensor(self.grid_size, device=self.device, dtype=torch.long) - 1
            
            grid_idx = torch.maximum(
                torch.minimum(grid_idx, max_grid_idx.unsqueeze(0).expand(self.num_envs, -1)), 
                torch.zeros_like(grid_idx)
            )
            env_indices = torch.arange(self.num_envs, device=self.device)

            prior_grid_values = self.grid_map[env_indices, grid_idx[:,0], grid_idx[:,1], grid_idx[:,2]]

            visit_reward = (1 - prior_grid_values.float()) * self.cfg.visit_reward_scale

            # Update the grid map to mark these positions as visited
            self.grid_map[env_indices, grid_idx[:,0], grid_idx[:,1], grid_idx[:,2]] = 1

            # Total reward
            rewards[agent] = progress_reward + collision_penalty + visit_reward
        return rewards

    def _get_dones(self):
        terminated = {}
        truncated = {}
        for agent in self.possible_agents:
            current_pos = self._robots[agent].data.root_pos_w - self.scene.env_origins
            out_of_bounds = (
                (current_pos < self.area_bounds["min"]) | (current_pos > self.area_bounds["max"])
            ).any(dim=-1)
            # TODO: Collision Detection
            # collision = self._robots[agent].data.contact_forces.norm(dim=-1) > 0.0
            collision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            max_episode_length = self.episode_length_buf >= self.max_episode_length

            terminated[agent] = out_of_bounds | collision
            truncated[agent] = max_episode_length
            self.reset_buf = terminated[agent] | truncated[agent]
        return terminated, truncated

    def _get_observations(self):
        observations = {}
        for agent in self.possible_agents:
            robot = self._robots[agent]
            own_pos = robot.data.root_pos_w - self.scene.env_origins
            # own_vel = robot.data.root_lin_vel_w

            # Other agents' positions
            other_agents_pos = []
            for other_agent in self.possible_agents:
                if other_agent != agent:
                    other_pos = self._robots[other_agent].data.root_pos_w - self.scene.env_origins
                    other_agents_pos.append(other_pos)
            other_agents_pos = torch.cat(other_agents_pos, dim=-1) if other_agents_pos else torch.zeros_like(own_pos)

            # Flatten grid map
            grid_map_flat = self.grid_map.view(self.num_envs, -1)

            obs = torch.cat([own_pos, other_agents_pos, grid_map_flat], dim=-1)
            observations[agent] = obs
        return observations