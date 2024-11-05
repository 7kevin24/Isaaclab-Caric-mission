from __future__ import annotations
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectMARLEnv, DirectMARLEnvCfg

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

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window."""
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    pass  # Add custom UI elements here


@configclass
class QuadcopterEnvCfg(DirectMARLEnvCfg):
    # Environment settings
    episode_length_s = 10.0
    decimation = 2

    # Agent settings
    num_agents: int = 2
    possible_agents = [f"agent_{i}" for i in range(num_agents)]

    # Define spaces using integers like Shadow Hand
    # Action space: [thrust, roll_moment, pitch_moment, yaw_moment]
    action_spaces = {agent: 4 for agent in possible_agents}

    # Observation space components (matching single-agent impl):
    # - root linear velocity in body frame (3)
    # - root angular velocity in body frame (3)
    # - projected gravity in body frame (3)
    # - desired position in body frame (3)
    observation_spaces = {agent: 12 for agent in possible_agents}

    state_space = 24  # 12 dims per agent for 2 agents
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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
        num_envs=2, 
        env_spacing=2.5, 
        replicate_physics=True
    )

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # Reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0

    # Robot configuration
    robot = {
        agent: CRAZYFLIE_CFG.replace(
            prim_path=f"/World/envs/env_.*/Robot_{agent}"
        ).replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),  # Default starting height
                rot=(1.0, 0.0, 0.0, 0.0),  # Default orientation
            )
        )
        for agent in possible_agents
    }


class QuadcopterEnv(DirectMARLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        self.possible_agents = cfg.possible_agents
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize per-agent tensors with correct shapes
        self._actions = {
            agent: torch.zeros(self.num_envs, cfg.action_spaces[agent], device=self.device)
            for agent in self.possible_agents
        }
        self._thrust = {
            agent: torch.zeros(self.num_envs, 1, 3, device=self.device)  # Note the [1] dimension
            for agent in self.possible_agents
        }
        self._moment = {
            agent: torch.zeros(self.num_envs, 1, 3, device=self.device)  # Note the [1] dimension
            for agent in self.possible_agents
        }
        self._desired_pos_w = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.possible_agents
        }

        # Logging
        self._episode_sums = {
            agent: {
                "lin_vel": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "ang_vel": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
                "distance_to_goal": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            }
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

        # Debug visualization
        # self.set_debug_vis(self.cfg.debug_vis)

    # Rest of your class methods...
    # Make sure not to override _configure_env_spaces unless necessary

    def _setup_scene(self):
        """Set up the scene following Shadow Hand's approach."""
        # Create terrain first
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Initialize robots dictionary
        self._robots = {}

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

    # Implement other necessary methods (_pre_physics_step, _apply_action, etc.)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        """Pre-process actions before stepping through physics."""
        for agent, action in actions.items():
            # Clamp actions to valid range
            self._actions[agent] = action.clone().clamp(-1.0, 1.0)
            
            # Reset thrust and moment tensors
            self._thrust[agent].zero_()
            
            # Compute thrust - shape [num_envs, 1, 3] to match single-agent implementation
            self._thrust[agent][:, 0, 2] = (
                self.cfg.thrust_to_weight 
                * self._robot_weights[agent] 
                * (self._actions[agent][:, 0] + 1.0) / 2.0
            )
            
            # Compute moment - shape [num_envs, 1, 3] to match single-agent implementation
            self._moment[agent][:, 0, :] = self.cfg.moment_scale * self._actions[agent][:, 1:]

    def _apply_action(self):
        """Apply actions as external forces and torques to the quadcopters."""
        for agent in self.possible_agents:
            self._robots[agent].set_external_force_and_torque(
                self._thrust[agent],  # [num_envs, 1, 3]
                self._moment[agent],  # [num_envs, 1, 3] 
                body_ids=self._body_ids[agent]
            )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations for each agent using body-frame quantities."""
        observations = {}
        for agent in self.possible_agents:
            robot = self._robots[agent]
            
            # Get desired position in body frame
            desired_pos_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[:, :3], 
                robot.data.root_state_w[:, 3:7], 
                self._desired_pos_w[agent]
            )
            
            # Combine observations (matching single-agent implementation)
            obs = torch.cat([
                robot.data.root_lin_vel_b,      # Linear velocity in body frame (3)
                robot.data.root_ang_vel_b,      # Angular velocity in body frame (3)
                robot.data.projected_gravity_b,  # Projected gravity in body frame (3)
                desired_pos_b,                   # Desired position in body frame (3)
            ], dim=-1)
            
            observations[agent] = obs
        
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Compute rewards with proper tensor types."""
        rewards = {}
        terminated_dict, _ = self._get_dones()  # Get termination flags
        
        for agent in self.possible_agents:
            # Basic rewards
            lin_vel = torch.norm(self._robots[agent].data.root_lin_vel_w, dim=1)
            ang_vel = torch.norm(self._robots[agent].data.root_ang_vel_w, dim=1)
            distance = torch.norm(self._desired_pos_w[agent] - self._robots[agent].data.root_pos_w, dim=1)

            # Collision penalty using boolean operations
            collision_penalty = torch.zeros(self.num_envs, device=self.device)
            for other_agent in self.possible_agents:
                if other_agent != agent:
                    distance_between_agents = torch.norm(
                        self._robots[agent].data.root_pos_w - self._robots[other_agent].data.root_pos_w, dim=1
                    )
                    collision = (distance_between_agents < 0.5).float()  # Convert boolean to float
                    collision_penalty += collision * 10.0

            # Compute reward with proper tensor types
            rewards[agent] = torch.where(
                terminated_dict[agent],
                torch.zeros_like(lin_vel),  # Zero reward on termination
                self.cfg.lin_vel_reward_scale * lin_vel +
                self.cfg.ang_vel_reward_scale * ang_vel +
                self.cfg.distance_to_goal_reward_scale * distance -
                collision_penalty
            )

        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute termination conditions.
        
        Returns:
            Tuple of dictionaries containing terminated and timeout flags for each agent
        """
        terminated_dict = {}
        timeout_dict = {}
        
        for agent in self.possible_agents:
            # Check position bounds as boolean
            pos = self._robots[agent].data.root_pos_w
            pos_terminated = torch.any(torch.abs(pos) > 5.0, dim=1)
            
            # Episode timeout as boolean
            timeout = self.episode_length_buf >= self.max_episode_length
            
            # Store as boolean tensors explicitly
            terminated_dict[agent] = pos_terminated.bool()
            timeout_dict[agent] = timeout.bool()
        
        return terminated_dict, timeout_dict

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments with proper origin handling."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Get terminated and timeout states before resetting
        terminated_dict, timeout_dict = self._get_dones()

        # Reset metrics logging
        for agent in self.possible_agents:
            # Compute final distance to goal for logging
            final_distance_to_goal = torch.linalg.norm(
                self._desired_pos_w[agent][env_ids] - self._robots[agent].data.root_pos_w[env_ids], dim=1
            ).mean()

            # Update extras with proper termination info
            self.extras[agent].update({
                "Metrics/final_distance_to_goal": final_distance_to_goal.item(),
                "Episode_Termination/died": torch.count_nonzero(terminated_dict[agent][env_ids]).item(),
                "Episode_Termination/time_out": torch.count_nonzero(timeout_dict[agent][env_ids]).item(),
            })

            # Reset episode sums
            for key in self._episode_sums[agent]:
                self._episode_sums[agent][key][env_ids] = 0.0

            # Reset robot state with proper env origins
            default_state = self._robots[agent].data.default_root_state.clone()[env_ids]
            
            # Add randomization to position
            pos_noise = torch.randn((len(env_ids), 3), device=self.device) * 0.5
            default_state[:, :3] = default_state[:, :3] + pos_noise + self.scene.env_origins[env_ids]
            
            # Write state back to simulation
            self._robots[agent].write_root_state_to_sim(default_state, env_ids)

            # Reset target positions with env origins
            self._desired_pos_w[agent][env_ids] = (
                torch.randn((len(env_ids), 3), device=self.device) * torch.tensor([2.0, 2.0, 0.5], device=self.device)
                + torch.tensor([0.0, 0.0, 1.5], device=self.device)
                + self.scene.env_origins[env_ids]
            )

        # Reset buffers
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0

        # Call parent reset to handle base functionality
        super()._reset_idx(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizers"):
                self.goal_pos_visualizers = {
                    agent: VisualizationMarkers(CUBOID_MARKER_CFG)
                    for agent in self.possible_agents
                }
            for agent in self.possible_agents:
                self.goal_pos_visualizers[agent].set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizers"):
                for agent in self.possible_agents:
                    self.goal_pos_visualizers[agent].set_visibility(False)

    def _debug_vis_callback(self, event):
        for agent in self.possible_agents:
            self.goal_pos_visualizers[agent].visualize(self._desired_pos_w[agent])

    def state(self) -> StateType | None:
        """Returns the state for centralized training."""
        if not self.cfg.state_space:
            return None
        
        # Concatenate all agent observations
        return torch.cat([
            self.obs_dict[agent] for agent in self.possible_agents
        ], dim=-1)
