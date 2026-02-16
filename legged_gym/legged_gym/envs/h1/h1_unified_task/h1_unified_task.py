import torch
import random
import numpy as np
import math
import os
import torch.nn.functional as F

from isaacgym.torch_utils import *
from legged_gym.envs.base.legged_robot import LeggedRobot, get_euler_xyz_tensor
from isaacgym import gymapi, gymtorch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import XBotTerrain
from collections import deque
from legged_gym.utils.human import sample_int_from_float, sample_rp


class H1UnifiedTask(LeggedRobot):
    '''
    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg

        self.hidden_z = cfg.asset.hidden_z

        self.num_tasks = self.cfg.env.num_tasks
        self.task_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        task_counts = [
            cfg.env.num_task_ball_envs,
            cfg.env.num_task_box_envs,
            cfg.env.num_task_button_envs,
            cfg.env.num_task_cabinet_envs,
            cfg.env.num_task_carry_envs,
            cfg.env.num_task_lift_envs,
            cfg.env.num_task_reach_envs,
            cfg.env.num_task_transfer_envs
        ]

        current_env_idx = 0
        for task_id, count in enumerate(task_counts):
            if count > 0:
                self.task_ids[current_env_idx : current_env_idx + count] = task_id
                current_env_idx += count

        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)

        # Task ball
        self.ori_ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        door_z_offsets = [offset[2] for offset in self.cfg.asset.door_offsets]
        self.door_z_offsets = torch.tensor(door_z_offsets, device=self.device)
        self.num_door_parts = len(self.cfg.asset.door_offsets)

        # Task box and transfer
        self.small_box_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Task button
        self.button_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        # Task cabinet
        self.cabinet_dof_goal = 0

        # Task carry and lift
        self.big_box_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.gym.simulate(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._init_target_wp()
        self.compute_observations()

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.humanoid_root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.humanoid_root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def  _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase
    
    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask

    def _init_target_wp(self):
        self.ori_wrist_pos = self.rigid_state[:, self.wrist_indices, :7].clone() # [num_envs, 2, 7], two hands
        self.target_wp, self.num_pairs, self.num_wp = sample_rp(self.device, num_points=2000000, num_wp=10, ranges=self.cfg.commands.ranges) # relative, self.target_wp.shape=[num_pairs, num_wp, 2, 7]
        self.target_wp_i = torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device) # for each env, choose one seq, [num_envs]
        self.target_wp_j = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # for each env, the timestep in the seq is initialized to 0, [num_envs]
        self.target_wp_dt = 1 / self.cfg.human.freq
        self.target_wp_update_steps = self.target_wp_dt / self.dt # not necessary integer
        assert self.dt <= self.target_wp_dt, f"self.dt {self.dt} must be less than self.target_wp_dt {self.target_wp_dt}"
        self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
        
        self.ref_dof_pos = None
        self.ref_wrist_pos = None
        self.ref_action = self.default_dof_pos
        self.delayed_obs_target_wp = None
        self.delayed_obs_target_wp_steps = self.cfg.human.delay / self.target_wp_dt
        self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        self.update_target_wp(torch.tensor([], dtype=torch.long, device=self.device))

    def update_target_wp(self, reset_env_ids):
        # self.target_wp_i specifies which seq to use for each env, and self.target_wp_j specifies the timestep in the seq
        self.ref_wrist_pos = self.target_wp[self.target_wp_i, self.target_wp_j] + self.ori_wrist_pos # [num_envs, 2, 7], two hands
        self.delayed_obs_target_wp = self.target_wp[self.target_wp_i, torch.maximum(self.target_wp_j - self.delayed_obs_target_wp_steps_int, torch.tensor(0))]
        resample_i = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.common_step_counter % self.target_wp_update_steps_int == 0:
            self.target_wp_j += 1
            wp_eps_end_bool = self.target_wp_j >= self.num_wp
            self.target_wp_j = torch.where(wp_eps_end_bool, torch.zeros_like(self.target_wp_j), self.target_wp_j)
            resample_i[wp_eps_end_bool.nonzero(as_tuple=False).flatten()] = True
            self.target_wp_update_steps_int = sample_int_from_float(self.target_wp_update_steps)
            self.delayed_obs_target_wp_steps_int = sample_int_from_float(self.delayed_obs_target_wp_steps)
        if self.cfg.human.resample_on_env_reset:
            self.target_wp_j[reset_env_ids] = 0
            resample_i[reset_env_ids] = True
        self.target_wp_i = torch.where(resample_i, torch.randint(0, self.num_pairs, (self.num_envs,), device=self.device), self.target_wp_i)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = XBotTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
        
    def _create_door_assets(self):
        door_assets = []
        door_pose = gymapi.Transform()

        for dim in self.cfg.asset.door_dims:
            door_dims = gymapi.Vec3(*dim)

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True

            door_asset = self.gym.create_box(
                self.sim,
                door_dims.x, door_dims.y, door_dims.z,
                asset_options
            )
            door_assets.append(door_asset)

        return door_assets, door_pose
    
    def _create_ball_asset(self):
        ball_size = self.cfg.asset.ball_size

        asset_options = gymapi.AssetOptions()
        ball_asset = self.gym.create_sphere(self.sim, ball_size, asset_options)

        ball_pose = gymapi.Transform()

        return ball_asset, ball_pose, ball_size
    
    def _create_front_table_asset(self):
        front_table_dims = gymapi.Vec3(*self.cfg.asset.front_table_dims)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        front_table_asset = self.gym.create_box(
            self.sim, front_table_dims.x, front_table_dims.y, front_table_dims.z, asset_options
        )

        front_table_pose = gymapi.Transform()

        return front_table_asset, front_table_pose, front_table_dims
    
    def _create_back_table_asset(self):
        back_table_dims = gymapi.Vec3(*self.cfg.asset.back_table_dims)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        back_table_asset = self.gym.create_box(
            self.sim, back_table_dims.x, back_table_dims.y, back_table_dims.z, asset_options
        )

        back_table_pose = gymapi.Transform()

        return back_table_asset, back_table_pose, back_table_dims
    
    def _create_small_box_asset(self):
        small_box_size = self.cfg.asset.small_box_size

        asset_options = gymapi.AssetOptions()
        small_box_asset = self.gym.create_box(self.sim, small_box_size, small_box_size, small_box_size, asset_options)

        small_box_pose = gymapi.Transform()

        return small_box_asset, small_box_pose, small_box_size
    
    def _create_big_box_asset(self):
        big_box_size = gymapi.Vec3(*self.cfg.asset.big_box_size)

        asset_options = gymapi.AssetOptions()
        big_box_asset = self.gym.create_box(self.sim, big_box_size.x, big_box_size.y, big_box_size.z, asset_options)

        big_box_pose = gymapi.Transform()

        return big_box_asset, big_box_pose, big_box_size
    
    def _create_wall_asset(self):
        wall_dims = gymapi.Vec3(*self.cfg.asset.wall_dims)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        wall_asset = self.gym.create_box(
            self.sim, wall_dims.x, wall_dims.y, wall_dims.z, asset_options
        )

        wall_pose = gymapi.Transform()

        return wall_asset, wall_pose, wall_dims
    
    def _create_cabinet_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_inertia = True
        asset_options.override_com = True
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        cabinet_asset = self.gym.load_asset(self.sim, self.cfg.asset.gapartnet_root, f"{self.cfg.asset.gapartnet_id}/mobility_annotation_gapartnet.urdf", asset_options)

        cabinet_pose = gymapi.Transform()

        # DOFs
        num_dofs = self.gym.get_asset_dof_count(cabinet_asset)
        dof_props = self.gym.get_asset_dof_properties(cabinet_asset)

        # Default DOF
        default_dof_pos = np.zeros(num_dofs, dtype=np.float32)
        default_dof_pos[:] = self.cfg.asset.cabinet_dof_default
        default_dof_state = np.zeros(num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        # Configure DOF properties
        dof_props["driveMode"].fill(gymapi.DOF_MODE_NONE) # NO DOF_MODE_POS
        dof_props["stiffness"].fill(0.0) # how fast the arti obj gonna move
        dof_props["damping"].fill(5.0) # large damping to prevent oscillation
        dof_props["friction"].fill(0.0)

        return cabinet_asset, cabinet_pose, num_dofs, dof_props, default_dof_state

    def _spawn_actor(self, env_handle, asset, pose, name, collision_group, collision_filter):
        h = self.gym.create_actor(env_handle, asset, pose, name, collision_group, collision_filter)
        idx = self.gym.get_actor_index(env_handle, h, gymapi.DOMAIN_SIM)
        return h, idx

    def _spawn_robot(self, env_handle, env_id, base_pos, robot_asset, start_pose, dof_props_asset, rigid_shape_props_asset):
        start_pose.p = gymapi.Vec3(
            base_pos[0].item(),
            base_pos[1].item(),
            base_pos[2].item(),
        )
        
        rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, env_id)
        self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

        robot_handle, robot_idx = self._spawn_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, env_id, self.cfg.asset.self_collisions)

        dof_props = self._process_dof_props(dof_props_asset, env_id)
        self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)

        body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
        body_props = self._process_rigid_body_props(body_props, env_id)
        self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)

        return robot_handle, robot_idx
    
    def _spawn_doors(self, env_handle, env_id, base_pos, door_assets, door_pose):
        env_door_idxs = []
        for door_i, door_asset in enumerate(door_assets):
            offset = self.cfg.asset.door_offsets[door_i]
            door_pose.p = gymapi.Vec3(
                base_pos[0].item() + offset[0],
                base_pos[1].item() + offset[1],
                base_pos[2].item() + offset[2],
            )

            _, idx = self._spawn_actor(env_handle, door_asset, door_pose, f"door_{door_i}", env_id, self.cfg.asset.self_collisions)
            env_door_idxs.append(idx)

        return env_door_idxs
    
    def _spawn_ball(self, env_handle, env_id, base_pos, ball_asset, ball_pose, ball_size):
            ball_pose.p.x = base_pos[0].item() + np.random.uniform(*self.cfg.asset.ball_range_x)
            ball_pose.p.y = base_pos[1].item() + np.random.uniform(*self.cfg.asset.ball_range_y)
            ball_pose.p.z = base_pos[2].item() + 0.5 * ball_size
            ball_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))

            ball_handle, ball_idx = self._spawn_actor(env_handle, ball_asset, ball_pose, "ball", env_id, self.cfg.asset.self_collisions)

            ball_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)

            for prop in ball_rigid_body_props:
                prop.mass = random.uniform(*self.cfg.asset.ball_range_mass)
            self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_rigid_body_props, recomputeInertia=True)

            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))

            self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            
            return ball_idx

    def _spawn_front_table(self, env_handle, env_id, base_pos, front_table_asset, front_table_pose):
        offset = self.cfg.asset.front_table_offset
        front_table_pose.p = gymapi.Vec3(
            base_pos[0].item() + offset[0],
            base_pos[1].item() + offset[1],
            base_pos[2].item() + offset[2],
        )

        _, idx = self._spawn_actor(env_handle, front_table_asset, front_table_pose, "front_table", env_id, self.cfg.asset.self_collisions)

        return idx, front_table_pose

    def _spawn_back_table(self, env_handle, env_id, base_pos, back_table_asset, back_table_pose):
        offset = self.cfg.asset.back_table_offset
        back_table_pose.p = gymapi.Vec3(
            base_pos[0].item() + offset[0],
            base_pos[1].item() + offset[1],
            base_pos[2].item() + offset[2],
        )

        _, idx = self._spawn_actor(env_handle, back_table_asset, back_table_pose, "back_table", env_id, self.cfg.asset.self_collisions)

        return idx, back_table_pose

    def _spawn_small_box_on_front_table(self, env_handle, env_id, front_table_pose, small_box_asset, small_box_pose, small_box_size):
        small_box_pose.p.x = front_table_pose.p.x + np.random.uniform(*self.cfg.asset.small_box_range_x)
        small_box_pose.p.y = front_table_pose.p.y + np.random.uniform(*self.cfg.asset.small_box_range_y)
        small_box_pose.p.z = front_table_pose.p.z + 0.5 * self.cfg.asset.front_table_dims[2] + 0.5 * small_box_size
        small_box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))

        box_handle, box_idx = self._spawn_actor(env_handle, small_box_asset, small_box_pose, "small_box", env_id, self.cfg.asset.self_collisions)

        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))

        self.gym.set_rigid_body_color(env_handle, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        return box_idx, box_handle

    def _spawn_big_box(self, env_handle, env_id, base_pos, big_box_asset, big_box_pose, big_box_size):
        offset_xy = self.cfg.asset.big_box_offset_xy
        big_box_pose.p = gymapi.Vec3(
            base_pos[0].item() + offset_xy[0] + np.random.uniform(*self.cfg.asset.big_box_range_x),
            base_pos[1].item() + offset_xy[1] + np.random.uniform(*self.cfg.asset.big_box_range_y),
            base_pos[2].item() + 0.5 * big_box_size.z,
        )
        
        h, idx = self._spawn_actor(env_handle, big_box_asset, big_box_pose, "big_box", env_id, self.cfg.asset.self_collisions)

        big_box_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, h)
        for prop in big_box_rigid_body_props:
            prop.mass = random.uniform(*self.cfg.asset.big_box_range_mass)
        self.gym.set_actor_rigid_body_properties(env_handle, h, big_box_rigid_body_props, recomputeInertia=True)

        big_box_rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, h)
        for prop in big_box_rigid_shape_props:
            prop.friction = 5.
        self.gym.set_actor_rigid_shape_properties(env_handle, h, big_box_rigid_shape_props)
        
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        self.gym.set_rigid_body_color(env_handle, h, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        
        return idx

    def _spawn_wall(self, env_handle, env_id, base_pos, wall_asset, wall_pose):
        offset = self.cfg.asset.wall_offset
        wall_pose.p = gymapi.Vec3(
            base_pos[0].item() + offset[0],
            base_pos[1].item() + offset[1],
            base_pos[2].item() + offset[2],
        )

        _, idx = self._spawn_actor(env_handle, wall_asset, wall_pose, "wall", env_id, self.cfg.asset.self_collisions)

        return idx

    def _spawn_cabinet(self, env_handle, env_id, base_pos, cabinet_asset, cabinet_pose, cabinet_dof_props, cabinet_default_dof_state):
        offset = self.cfg.asset.cabinet_offset
        cabinet_pose.p = gymapi.Vec3(
            base_pos[0].item() + offset[0],
            base_pos[1].item() + offset[1],
            base_pos[2].item() + offset[2],
        )

        cabinet_handle, cabinet_idx = self._spawn_actor(env_handle, cabinet_asset, cabinet_pose, "cabinet", env_id, self.cfg.asset.self_collisions)

        self.gym.set_actor_dof_properties(env_handle, cabinet_handle, cabinet_dof_props)
        self.gym.set_actor_dof_states(env_handle, cabinet_handle, cabinet_default_dof_state, gymapi.STATE_ALL)
        self.gym.set_actor_scale(env_handle, cabinet_handle, self.cfg.asset.cabinet_scale)

        return cabinet_idx, cabinet_handle

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        knee_names = [s for s in self.body_names if self.cfg.asset.knee_name in s]
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # Create assets
        door_assets, door_pose = self._create_door_assets()
        self.door_idxs = []
        ball_asset, ball_pose, ball_size = self._create_ball_asset()
        self.ball_idxs = []
        front_table_asset, front_table_pose, front_table_dims = self._create_front_table_asset()
        self.front_table_idxs = []
        back_table_asset, back_table_pose, back_table_dims = self._create_back_table_asset()
        self.back_table_idxs = []
        small_box_asset, small_box_pose, small_box_size = self._create_small_box_asset()
        self.small_box_idxs = []
        big_box_asset, big_box_pose, big_box_size = self._create_big_box_asset()
        self.big_box_idxs = []  
        wall_asset, wall_pose, wall_dims = self._create_wall_asset()
        self.wall_idxs = []
        cabinet_asset, cabinet_pose, cabinet_num_dofs, cabinet_dof_props, cabinet_default_dof_state = self._create_cabinet_asset()
        self.cabinet_num_dofs = cabinet_num_dofs
        self.cabinet_idxs = []
        self.humanoid_idxs = []

        # Create actors
        for i in range(self.num_envs):
            ## Create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            pos = self.env_origins[i].clone()

            robot_handle, robot_idx = self._spawn_robot(env_handle, i, pos, robot_asset, start_pose, dof_props_asset, rigid_shape_props_asset)
            self.actor_handles.append(robot_handle)
            self.humanoid_idxs.append(robot_idx)

            env_door_idxs = self._spawn_doors(env_handle, i, pos, door_assets, door_pose)
            self.door_idxs.append(env_door_idxs)

            ball_idx = self._spawn_ball(env_handle, i, pos, ball_asset, ball_pose, ball_size)
            self.ball_idxs.append(ball_idx)

            front_table_idx, current_front_table_pose = self._spawn_front_table(env_handle, i, pos, front_table_asset, front_table_pose)
            self.front_table_idxs.append(front_table_idx)

            back_table_idx, _ = self._spawn_back_table(env_handle, i, pos, back_table_asset, back_table_pose)
            self.back_table_idxs.append(back_table_idx)

            small_box_idx, _ = self._spawn_small_box_on_front_table(env_handle, i, current_front_table_pose, small_box_asset, small_box_pose, small_box_size)
            self.small_box_idxs.append(small_box_idx)

            big_box_idx = self._spawn_big_box(env_handle, i, pos, big_box_asset, big_box_pose, big_box_size)
            self.big_box_idxs.append(big_box_idx)

            wall_idx = self._spawn_wall(env_handle, i, pos, wall_asset, wall_pose)
            self.wall_idxs.append(wall_idx)

            cabinet_idx, _ = self._spawn_cabinet(env_handle, i, pos, cabinet_asset, cabinet_pose, cabinet_dof_props, cabinet_default_dof_state)
            self.cabinet_idxs.append(cabinet_idx)

        self._create_sensors_all()
        self.humanoid_idxs = torch.tensor(self.humanoid_idxs, device=self.device)
        self.door_idxs = torch.tensor(self.door_idxs, device=self.device)
        self.ball_idxs = torch.tensor(self.ball_idxs, device=self.device)
        self.front_table_idxs = torch.tensor(self.front_table_idxs, device=self.device)
        self.back_table_idxs = torch.tensor(self.back_table_idxs, device=self.device)
        self.small_box_idxs = torch.tensor(self.small_box_idxs, device=self.device)
        self.big_box_idxs = torch.tensor(self.big_box_idxs, device=self.device)
        self.wall_idxs = torch.tensor(self.wall_idxs, device=self.device)
        self.cabinet_idxs = torch.tensor(self.cabinet_idxs, device=self.device)

        ### Common body parts
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        ### Other body parts
        #### Elbow
        elbow_names = [s for s in self.body_names if self.cfg.asset.elbow_name in s]
        self.elbow_indices = torch.zeros(len(elbow_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(elbow_names)):
            self.elbow_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], elbow_names[i])
        #### Torso 
        torso_names = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(torso_names)):
            self.torso_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], torso_names[i])
        #### Wrist
        wrist_names = [s for s in self.body_names if self.cfg.asset.wrist_name in s]
        self.wrist_indices = torch.zeros(len(wrist_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wrist_names)):
            self.wrist_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], wrist_names[i])
        
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self._init_visual_buffers()
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        self.humanoid_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.humanoid_idxs[0]]
        
        self.ball_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.ball_idxs[0]]
        self.front_table_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.front_table_idxs[0]]
        self.back_table_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.back_table_idxs[0]]
        self.small_box_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.small_box_idxs[0]]
        self.big_box_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.big_box_idxs[0]]
        self.wall_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.wall_idxs[0]]
        self.cabinet_root_states = self.root_states.view(self.num_envs, -1, 13)[:, self.cabinet_idxs[0]]
        
        # Task cabinet
        self.humanoid_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof]
        self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof:]

        self.dof_pos = self.humanoid_dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.humanoid_dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        self.base_quat = self.humanoid_root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_rigid_state = torch.zeros_like(self.rigid_state)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.humanoid_root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            # print(name)
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():

                if dof_name in name:
                    self.p_gains[:, i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:, i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:, i] = 0.
                self.d_gains[:, i] = 0.
                print(f"PD gain of joint {name} were not defined, setting them to zero")
        

        self.rand_push_force = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.rand_push_torque = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_joint_pd_target = self.default_dof_pos.clone()
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))
            
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        ## Task cabinet
        self._reset_cabinet_dofs(env_ids)

        humanoid_ids_int32 = self.humanoid_idxs[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(humanoid_ids_int32), len(humanoid_ids_int32))
        
        ## Task cabinet
        cabinet_ids_int32 = self.cabinet_idxs[env_ids].to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(cabinet_ids_int32), len(cabinet_ids_int32))
        
    def _reset_cabinet_dofs(self, env_ids):
        active_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_CABINET)
        active_ids = env_ids[active_mask]

        if len(active_ids) == 0:
            return
        
        self.cabinet_dof_state[active_ids, :, 0] = self.cfg.asset.cabinet_dof_default
        self.cabinet_dof_state[active_ids, :, 1] = 0.0
        self.cabinet_dof_goal = 0

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.humanoid_root_states[env_ids] = self.base_init_state
            self.humanoid_root_states[env_ids, :3] += self.env_origins[env_ids]
            self.humanoid_root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.humanoid_root_states[env_ids] = self.base_init_state
            self.humanoid_root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.humanoid_root_states[env_ids, 7:13] = torch_rand_float(-0.05, 0.05, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.asset.fix_base_link:
            self.humanoid_root_states[env_ids, 7:13] = 0
            self.humanoid_root_states[env_ids, 2] += 1.8
        
        self._hide_all_assets(env_ids)

        self._reset_door_states(env_ids)
        self._reset_ball_states(env_ids)
        self._reset_front_table_states(env_ids)
        self._reset_back_table_states(env_ids)
        self._reset_small_box_states(env_ids)
        self._reset_big_box_states(env_ids)
        self._reset_wall_states(env_ids)
        self._reset_cabinet_states(env_ids)
        
        humanoid_ids_int32 = self.humanoid_idxs[env_ids].to(torch.int32)
        ball_ids_int32 = self.ball_idxs[env_ids].to(torch.int32)
        door_ids_int32 = self.door_idxs[env_ids].flatten().to(torch.int32)
        front_table_ids_int32 = self.front_table_idxs[env_ids].to(torch.int32)
        back_table_ids_int32 = self.back_table_idxs[env_ids].to(torch.int32)
        small_box_ids_int32 = self.small_box_idxs[env_ids].to(torch.int32)
        big_box_ids_int32 = self.big_box_idxs[env_ids].to(torch.int32)
        wall_ids_int32 = self.wall_idxs[env_ids].to(torch.int32)
        cabinet_ids_int32 = self.cabinet_idxs[env_ids].to(torch.int32)
        
        all_actor_indices = torch.cat([
            humanoid_ids_int32,
            ball_ids_int32,
            door_ids_int32,
            front_table_ids_int32,
            back_table_ids_int32,
            small_box_ids_int32,
            big_box_ids_int32,
            wall_ids_int32,
            cabinet_ids_int32,
        ])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(all_actor_indices),
            len(all_actor_indices)
        )

    def _hide_all_assets(self, env_ids):
        pos = self.env_origins[env_ids].clone()

        self.ball_root_states[env_ids, 0] = pos[:, 0] - 10
        self.ball_root_states[env_ids, 1] = pos[:, 1] - 10
        self.ball_root_states[env_ids, 2] = self.hidden_z * 3
        self.ball_root_states[env_ids, 7:13] = 0

        door_actor_ids = self.door_idxs[env_ids].flatten()
        self.root_states[door_actor_ids, 2] = self.hidden_z
        self.root_states[door_actor_ids, 7:13] = 0

        self.front_table_root_states[env_ids, 2] = self.hidden_z
        self.front_table_root_states[env_ids, 7:13] = 0

        self.back_table_root_states[env_ids, 2] = self.hidden_z
        self.back_table_root_states[env_ids, 7:13] = 0

        self.small_box_root_states[env_ids, 0] = pos[:, 0] - 15
        self.small_box_root_states[env_ids, 1] = pos[:, 1] - 15
        self.small_box_root_states[env_ids, 2] = self.hidden_z * 3
        self.small_box_root_states[env_ids, 7:13] = 0

        self.big_box_root_states[env_ids, 0] = pos[:, 0] - 20
        self.big_box_root_states[env_ids, 1] = pos[:, 1] - 20
        self.big_box_root_states[env_ids, 2] = self.hidden_z * 3
        self.big_box_root_states[env_ids, 7:13] = 0

        self.wall_root_states[env_ids, 0] = pos[:, 0]
        self.wall_root_states[env_ids, 1] = pos[:, 1]
        self.wall_root_states[env_ids, 2] = self.hidden_z
        self.wall_root_states[env_ids, 7:13] = 0

        self.cabinet_root_states[env_ids, 2] = self.hidden_z * 1.1
        self.cabinet_root_states[env_ids, 7:13] = 0

    def _reset_door_states(self, env_ids):
        active_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_BALL)
        active_ids = env_ids[active_mask]

        if len(active_ids) == 0:
            return
        
        pos = self.env_origins[active_ids].clone()

        active_door_indices = self.door_idxs[active_ids].flatten()

        target_z_offsets = self.door_z_offsets.repeat(len(active_ids))

        env_base_z = pos[:, 2].repeat_interleave(self.num_door_parts)
        final_z = env_base_z + target_z_offsets

        self.root_states[active_door_indices, 2] = final_z
        self.root_states[active_door_indices, 7:13] = 0.0

    def _reset_ball_states(self, env_ids):
        active_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_BALL)
        active_ids = env_ids[active_mask]

        if len(active_ids) == 0:
            return
        
        pos = self.env_origins[active_ids].clone()

        self.ori_ball_pos[active_ids, 0] = pos[:, 0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.ball_range_x).to(self.device)
        self.ori_ball_pos[active_ids, 1] = pos[:, 1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.ball_range_y).to(self.device)
        self.ori_ball_pos[active_ids, 2] = 0.5 * self.cfg.asset.ball_size

        self.ball_root_states[active_ids, :3] = self.ori_ball_pos[active_ids].clone()
        self.ball_root_states[active_ids, 3] = 1
        self.ball_root_states[active_ids, 4:] = 0

    def _reset_front_table_states(self, env_ids):
        active_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_BOX) | (self.task_ids[env_ids] == self.cfg.env.TASK_TRANSFER)
        active_ids = env_ids[active_mask]

        if len(active_ids) == 0:
            return
        
        pos = self.env_origins[active_ids].clone()

        self.front_table_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.front_table_offset[0]
        self.front_table_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.front_table_offset[1]
        self.front_table_root_states[active_ids, 2] = pos[:, 2] + self.cfg.asset.front_table_offset[2]
        self.front_table_root_states[active_ids, 7:13] = 0

    def _reset_back_table_states(self, env_ids):
        active_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_TRANSFER)
        active_ids = env_ids[active_mask]

        if len(active_ids) == 0:
            return
        
        pos = self.env_origins[active_ids].clone()

        self.back_table_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.back_table_offset[0]
        self.back_table_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.back_table_offset[1]
        self.back_table_root_states[active_ids, 2] = pos[:, 2] + self.cfg.asset.back_table_offset[2]
        self.back_table_root_states[active_ids, 7:13] = 0

    def _reset_small_box_states(self, env_ids):
        active_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_BOX) | (self.task_ids[env_ids] == self.cfg.env.TASK_TRANSFER)
        active_ids = env_ids[active_mask]

        if len(active_ids) == 0:
            return

        self.small_box_root_states[active_ids, 0] = self.front_table_root_states[active_ids, 0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.small_box_range_x).to(self.device)
        self.small_box_root_states[active_ids, 1] = self.front_table_root_states[active_ids, 1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.small_box_range_y).to(self.device)
        self.small_box_root_states[active_ids, 2] = self.front_table_root_states[active_ids, 2] + 0.5 * self.cfg.asset.front_table_dims[2] + 0.5 * self.cfg.asset.small_box_size
        self.small_box_root_states[active_ids, 3] = 1
        self.small_box_root_states[active_ids, 4:] = 0

    def _reset_big_box_states(self, env_ids):
        active_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_CARRY) | (self.task_ids[env_ids] == self.cfg.env.TASK_LIFT)
        active_ids = env_ids[active_mask]

        if len(active_ids) == 0:
            return

        pos = self.env_origins[active_ids].clone()

        self.big_box_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.big_box_offset_xy[0] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.big_box_range_x).to(self.device)
        self.big_box_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.big_box_offset_xy[1] + torch.FloatTensor(len(active_ids)).uniform_(*self.cfg.asset.big_box_range_y).to(self.device)
        self.big_box_root_states[active_ids, 2] = 0.5 * self.cfg.asset.big_box_size[2]
        self.big_box_root_states[active_ids, 3] = 1
        self.big_box_root_states[active_ids, 4:] = 0

    def _reset_wall_states(self, env_ids):
        active_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_BUTTON)
        active_ids = env_ids[active_mask]

        if len(active_ids) == 0:
            return
        
        pos = self.env_origins[active_ids].clone()

        self.wall_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.wall_offset[0]
        self.wall_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.wall_offset[1]
        self.wall_root_states[active_ids, 2] = pos[:, 2] + self.cfg.asset.wall_offset[2]
        self.wall_root_states[active_ids, 7:13] = 0

    def _reset_cabinet_states(self, env_ids):
        active_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_CABINET)
        active_ids = env_ids[active_mask]

        if len(active_ids) == 0:
            return
        
        pos = self.env_origins[active_ids].clone()

        self.cabinet_root_states[active_ids, 0] = pos[:, 0] + self.cfg.asset.cabinet_offset[0]
        self.cabinet_root_states[active_ids, 1] = pos[:, 1] + self.cfg.asset.cabinet_offset[1]
        self.cabinet_root_states[active_ids, 2] = pos[:, 2] + self.cfg.asset.cabinet_offset[2]
        self.cabinet_root_states[active_ids, 7:13] = 0.0

    def step(self, actions):
        with torch.no_grad():
            actions = actions.to(self.device)
            actions = actions.detach()

            if self.cfg.env.use_ref_actions:
                actions += self.ref_action
            # dynamic randomization
            # delay = torch.rand((self.num_envs, 1), device=self.device)
            delay = torch.rand((self.num_envs, 1), device=self.device)
            actions = (1 - delay) * actions.to(self.device) + delay * self.actions
            actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
            
            # changed version of super().step()
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
            # step physics and render each frame
            self.render()
            for _ in range(self.cfg.control.decimation):
                self.torques = self._compute_torques(self.actions).view(self.torques.shape) # [num_envs, num_actions]
                cabinet_force_buffer = torch.zeros((self.num_envs, self.cabinet_num_dofs), device=self.device)
                full_force_buffer = torch.cat((self.torques, cabinet_force_buffer), dim=1) # [num_envs, num_dofs + cabinet_num_dofs]
                humanoid_ids_int32 = self.humanoid_idxs.to(dtype=torch.int32)
                self.gym.set_dof_actuation_force_tensor_indexed(self.sim, 
                                                                gymtorch.unwrap_tensor(full_force_buffer),
                                                                gymtorch.unwrap_tensor(humanoid_ids_int32), len(humanoid_ids_int32))

                self.gym.simulate(self.sim)
                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
            self.post_physics_step()

            # return clipped obs, clipped states (None), rewards, dones and infos
            clip_obs = self.cfg.normalization.clip_observations
            self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
            if self.privileged_obs_buf is not None:
                self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.humanoid_root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.humanoid_root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.humanoid_root_states[:, 7:13]
        self.last_rigid_state[:] = self.rigid_state[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def compute_observations(self):
        self.compute_visual_observations()
        
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        self.command_input_wo_clock = self.commands[:, :3] * self.commands_scale

        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        task_one_hot = F.one_hot(self.task_ids, num_classes=self.num_tasks).float()

        common_obs_buf = torch.cat((
            q,    # |A|
            dq,  # |A|
            self.actions,   # |A|
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            task_one_hot, # 8
        ), dim=-1)

        common_privileged_obs_buf = torch.cat((
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # |A|
            self.dof_vel * self.obs_scales.dof_vel,  # |A|
            self.actions,  # |A|
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 2
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1
            # stance_mask,  # 2
            contact_mask,  # 2
            task_one_hot, # 8
        ), dim=-1)

        task_specific_obs_buf = torch.zeros(self.num_envs, self.cfg.env.command_dim, device=self.device)
        task_specific_privileged_obs_buf = torch.zeros(self.num_envs, self.cfg.env.max_privileged_obs_dim, device=self.device)

        # Task ball observations
        task_ball_mask = (self.task_ids == self.cfg.env.TASK_BALL)
        if torch.any(task_ball_mask):
            ball_pos = self.ball_root_states[task_ball_mask, :3]
            torso_pos = self.rigid_state[task_ball_mask, self.torso_indices, :3].squeeze(1)
            ball_goal_diff = ball_pos - self.ball_goal_pos[task_ball_mask]
            root_ball_diff = torso_pos - ball_pos

            goal_pos_obs = torch.flatten(self.ball_goal_pos[task_ball_mask], start_dim=1) # [num_envs, 3]
            ball_pos_obs = torch.flatten(ball_pos, start_dim=1) # [num_envs, 3]
            torso_pos_obs = torch.flatten(torso_pos, start_dim=1) # [num_envs, 3]
            ball_goal_diff_obs = torch.flatten(ball_goal_diff, start_dim=1) # [num_envs, 3]
            root_ball_diff_obs = torch.flatten(root_ball_diff, start_dim=1) # [num_envs, 3]

            task_ball_obs = torch.cat((ball_goal_diff_obs, root_ball_diff_obs), dim=-1)
            task_specific_obs_buf[task_ball_mask, :6] = task_ball_obs

            task_ball_privileged_obs = torch.cat((
                goal_pos_obs, # 3
                ball_pos_obs, # 3
                torso_pos_obs, # 3
                ball_goal_diff_obs,  # 3
                root_ball_diff_obs,  # 3
            ), dim=-1)
            task_specific_privileged_obs_buf[task_ball_mask, :15] = task_ball_privileged_obs

        # Task box and transfer observations
        task_box_and_transfer_mask = (self.task_ids == self.cfg.env.TASK_BOX) | (self.task_ids == self.cfg.env.TASK_TRANSFER)
        if torch.any(task_box_and_transfer_mask):
            wrist_pos = self.rigid_state[task_box_and_transfer_mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            small_box_pos = self.small_box_root_states[task_box_and_transfer_mask, :3]
            diff = small_box_pos - self.small_box_goal_pos[task_box_and_transfer_mask]
            wrist_box_diff = wrist_pos - small_box_pos.unsqueeze(1) # [num_envs, 2, 3], two hands, position only
            
            wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1) # [num_envs, 6]
            wrist_box_diff_obs = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
            box_goal_pos_obs = torch.flatten(self.small_box_goal_pos[task_box_and_transfer_mask], start_dim=1) # [num_envs, 3]
            small_box_pos_obs = torch.flatten(small_box_pos, start_dim=1) # [num_envs, 3]
            diff_obs = torch.flatten(diff, start_dim=1) # [num_envs, 3]

            task_box_and_transfer_obs = torch.cat((diff_obs, wrist_box_diff_obs), dim=-1)
            task_specific_obs_buf[task_box_and_transfer_mask, :9] = task_box_and_transfer_obs

            task_box_privileged_obs = torch.cat((
                box_goal_pos_obs, # 3
                small_box_pos_obs, # 3
                diff_obs,  # 3
                wrist_pos_obs, # 6
                wrist_box_diff_obs, # 6
            ), dim=-1)
            task_specific_privileged_obs_buf[task_box_and_transfer_mask, :21] = task_box_privileged_obs

        # Task button observations
        task_button_mask = (self.task_ids == self.cfg.env.TASK_BUTTON)
        if torch.any(task_button_mask):
            wrist_pos = self.rigid_state[task_button_mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:, 0, :3] # [num_envs, 3], left hand, position only
            button_goal_pos = self.button_goal_pos[task_button_mask, :3] # [num_envs, 3]
            wrist_button_diff = wrist_pos - button_goal_pos # [num_envs, 3], left hand, position only

            task_button_obs = wrist_button_diff
            task_specific_obs_buf[task_button_mask, :3] = task_button_obs

            task_button_privileged_obs = torch.cat((
                button_goal_pos, # 3
                wrist_pos, # 3
                wrist_button_diff, # 3
            ), dim=-1)
            task_specific_privileged_obs_buf[task_button_mask, :9] = task_button_privileged_obs

        # Task cabinet observations
        task_cabinet_mask = (self.task_ids == self.cfg.env.TASK_CABINET)
        if torch.any(task_cabinet_mask):
            wrist_pos = self.rigid_state[task_cabinet_mask][:, self.wrist_indices, :3] # [num_envs, 2, 3], two hands
            # torso_pos = self.rigid_state[task_cabinet_mask, self.torso_indices, :3].squeeze(1) # [num_envs, 3]
            cabinet_pos = self.cabinet_root_states[task_cabinet_mask, :3] # [num_envs, 3]
            cabinet_dof_pos = self.cabinet_dof_state[task_cabinet_mask][:, :, 0] # [num_envs, 2]
            cabinet_dof_goal = self.cabinet_dof_goal # 0
            cabinet_dof_diff_obs = cabinet_dof_pos - cabinet_dof_goal # [num_envs, 2]
            wrist_cabinet_diff = wrist_pos - cabinet_pos.unsqueeze(1) # [num_envs, 2, 3]
            wrist_cabinet_diff_obs = torch.flatten(wrist_cabinet_diff, start_dim=1) # [num_envs, 6]

            task_cabinet_obs = torch.cat((cabinet_dof_diff_obs, wrist_cabinet_diff_obs), dim=-1)
            task_specific_obs_buf[task_cabinet_mask, :8] = task_cabinet_obs

            task_cabinet_privileged_obs = torch.cat((
                cabinet_dof_diff_obs, # 2
                wrist_cabinet_diff_obs, # 6
            ), dim=-1)
            task_specific_privileged_obs_buf[task_cabinet_mask, :8] = task_cabinet_privileged_obs

        # Task carry and lift observations
        task_carry_and_lift_mask = (self.task_ids == self.cfg.env.TASK_CARRY) | (self.task_ids == self.cfg.env.TASK_LIFT)
        if torch.any(task_carry_and_lift_mask):
            wrist_pos = self.rigid_state[task_carry_and_lift_mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            big_box_pos = self.big_box_root_states[task_carry_and_lift_mask, :3]
            diff = big_box_pos - self.big_box_goal_pos[task_carry_and_lift_mask]
            wrist_box_diff = wrist_pos - big_box_pos.unsqueeze(1) # [num_envs, 2, 3], two hands, position only
            
            wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1) # [num_envs, 6]
            wrist_box_diff_obs = torch.flatten(wrist_box_diff, start_dim=1) # [num_envs, 6]
            box_goal_pos_obs = torch.flatten(self.big_box_goal_pos[task_carry_and_lift_mask], start_dim=1) # [num_envs, 3]
            big_box_pos_obs = torch.flatten(big_box_pos, start_dim=1) # [num_envs, 3]
            diff_obs = torch.flatten(diff, start_dim=1) # [num_envs, 3]

            task_carry_obs = torch.cat((diff_obs, wrist_box_diff_obs), dim=-1)
            task_specific_obs_buf[task_carry_and_lift_mask, :9] = task_carry_obs

            task_carry_privileged_obs = torch.cat((
                box_goal_pos_obs, # 3
                big_box_pos_obs, # 3
                diff_obs,  # 3
                wrist_pos_obs, # 6
                wrist_box_diff_obs, # 6
            ), dim=-1)
            task_specific_privileged_obs_buf[task_carry_and_lift_mask, :21] = task_carry_privileged_obs

        # Task reach observations
        task_reach_mask = (self.task_ids == self.cfg.env.TASK_REACH)
        if torch.any(task_reach_mask):
            wrist_pos = self.rigid_state[task_reach_mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            diff = wrist_pos - self.ref_wrist_pos[task_reach_mask] # [num_envs, 2, 7], two hands

            ref_wrist_pos_obs = torch.flatten(self.ref_wrist_pos[task_reach_mask], start_dim=1) # [num_envs, 14]
            wrist_pos_obs = torch.flatten(wrist_pos, start_dim=1) # [num_envs, 14]
            diff_obs = torch.flatten(diff, start_dim=1) # [num_envs, 14]

            task_reach_obs = diff_obs
            task_specific_obs_buf[task_reach_mask, :14] = task_reach_obs

            task_reach_privileged_obs = torch.cat((
                ref_wrist_pos_obs,  # 14
                wrist_pos_obs,  # 14
                diff_obs,  # 14
            ), dim=-1)
            task_specific_privileged_obs_buf[task_reach_mask, :42] = task_reach_privileged_obs

        # Concat
        obs_buf = torch.cat((task_specific_obs_buf, common_obs_buf), dim=-1)
        self.privileged_obs_buf = torch.cat((task_specific_privileged_obs_buf, common_privileged_obs_buf), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.humanoid_root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

        is_ball_task = (self.task_ids == self.cfg.env.TASK_BALL)
        ball_pos = self.ball_root_states[:, :3]
        ball_goal_dist = torch.norm(ball_pos - self.ball_goal_pos, dim=1)

        self.reset_buf |= (is_ball_task & (ball_goal_dist < self.cfg.commands.ranges.ball_threshold))

    def _sample_goals(self, env_ids):
        ball_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_BALL)
        if torch.any(ball_mask):
            ball_env_ids = env_ids[ball_mask]
            pos = self.env_origins[ball_env_ids]
            self.ball_goal_pos[ball_env_ids, 0] = pos[:, 0] + torch.FloatTensor(len(ball_env_ids)).uniform_(*self.cfg.commands.ranges.ball_goal_x).to(self.device)
            self.ball_goal_pos[ball_env_ids, 1] = pos[:, 1] + torch.FloatTensor(len(ball_env_ids)).uniform_(*self.cfg.commands.ranges.ball_goal_y).to(self.device)
            self.ball_goal_pos[ball_env_ids, 2] = torch.FloatTensor(len(ball_env_ids)).uniform_(*self.cfg.commands.ranges.ball_goal_z).to(self.device)

        box_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_BOX)
        if torch.any(box_mask):
            box_env_ids = env_ids[box_mask]
            pos = self.env_origins[box_env_ids]
            self.small_box_goal_pos[box_env_ids, 0] = self.small_box_root_states[box_env_ids, 0] + torch.FloatTensor(len(box_env_ids)).uniform_(*self.cfg.commands.ranges.small_box_pos_x).to(self.device)
            self.small_box_goal_pos[box_env_ids, 1] = self.small_box_root_states[box_env_ids, 1] + torch.FloatTensor(len(box_env_ids)).uniform_(*self.cfg.commands.ranges.small_box_pos_y).to(self.device)
            self.small_box_goal_pos[box_env_ids, 2] = self.small_box_root_states[box_env_ids, 2].clone()

        button_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_BUTTON)
        if torch.any(button_mask):
            button_env_ids = env_ids[button_mask]
            pos = self.env_origins[button_env_ids]
            self.button_goal_pos[button_env_ids, 0] = self.wall_root_states[button_env_ids, 0].clone()
            self.button_goal_pos[button_env_ids, 1] = self.wall_root_states[button_env_ids, 1] + torch.FloatTensor(len(button_env_ids)).uniform_(*self.cfg.commands.ranges.button_pos_y).to(self.device)
            self.button_goal_pos[button_env_ids, 2] = self.cfg.asset.button_ori_z + torch.FloatTensor(len(button_env_ids)).uniform_(*self.cfg.commands.ranges.button_pos_z).to(self.device)

        carry_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_CARRY)
        if torch.any(carry_mask):
            carry_env_ids = env_ids[carry_mask]
            pos = self.env_origins[carry_env_ids]
            self.big_box_goal_pos[carry_env_ids, 0] = pos[:, 0] + torch.FloatTensor(len(carry_env_ids)).uniform_(*self.cfg.commands.ranges.big_box_pos_x).to(self.device)
            self.big_box_goal_pos[carry_env_ids, 1] = pos[:, 1] + torch.FloatTensor(len(carry_env_ids)).uniform_(*self.cfg.commands.ranges.big_box_pos_y).to(self.device)
            self.big_box_goal_pos[carry_env_ids, 2] = self.big_box_root_states[carry_env_ids, 2].clone()

        lift_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_LIFT)
        if torch.any(lift_mask):
            lift_env_ids = env_ids[lift_mask]
            pos = self.env_origins[lift_env_ids]
            self.big_box_goal_pos[lift_env_ids, 0] = self.big_box_root_states[lift_env_ids, 0].clone()
            self.big_box_goal_pos[lift_env_ids, 1] = self.big_box_root_states[lift_env_ids, 1].clone()
            self.big_box_goal_pos[lift_env_ids, 2] = self.big_box_root_states[lift_env_ids, 2] + torch.FloatTensor(len(lift_env_ids)).uniform_(*self.cfg.commands.ranges.big_box_pos_z).to(self.device)

        transfer_mask = (self.task_ids[env_ids] == self.cfg.env.TASK_TRANSFER)
        if torch.any(transfer_mask):
            transfer_env_ids = env_ids[transfer_mask]
            pos = self.env_origins[transfer_env_ids]
            self.small_box_goal_pos[transfer_env_ids, 0] = self.back_table_root_states[transfer_env_ids, 0] + torch.FloatTensor(len(transfer_env_ids)).uniform_(*self.cfg.asset.small_box_range_x).to(self.device)
            self.small_box_goal_pos[transfer_env_ids, 1] = self.back_table_root_states[transfer_env_ids, 1] + torch.FloatTensor(len(transfer_env_ids)).uniform_(*self.cfg.asset.small_box_range_y).to(self.device)
            self.small_box_goal_pos[transfer_env_ids, 2] = self.small_box_root_states[transfer_env_ids, 2]
        
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._sample_goals(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ================================================ Rewards ================================================== #

    def _reward_torso_ori_ball_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_BALL)

        if torch.any(mask):
            torso_pos = self.rigid_state[mask][:, self.torso_indices, :3].squeeze(1) # [envs, 3]
            torso_ori_ball_pos_diff = self.ori_ball_pos[mask] - torso_pos
            torso_ori_ball_pos_diff = torso_ori_ball_pos_diff[:, :2] # only xy
            torso_ori_ball_pos_error = torch.mean(torch.abs(torso_ori_ball_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * torso_ori_ball_pos_error)
            error[mask] = torso_ori_ball_pos_error

        return reward, error, mask
    
    def _reward_ball_goal_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_BALL)

        if torch.any(mask):
            ball_goal_diff = self.ball_root_states[mask, :3] - self.ball_goal_pos[mask]
            ball_goal_error = torch.mean(torch.abs(ball_goal_diff), dim=1)

            reward[mask] = torch.exp(-1 * ball_goal_error)
            error[mask] = ball_goal_error

        return reward, error, mask

    def _reward_small_box_goal_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_BOX) | (self.task_ids == self.cfg.env.TASK_TRANSFER)

        if torch.any(mask):
            small_box_pos_diff = self.small_box_root_states[mask, :3] - self.small_box_goal_pos[mask]
            small_box_pos_error = torch.mean(torch.abs(small_box_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * small_box_pos_error)
            error[mask] = small_box_pos_error

        return reward, error, mask

    def _reward_wrist_small_box_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_BOX) | (self.task_ids == self.cfg.env.TASK_TRANSFER)

        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            small_box_pos = self.small_box_root_states[mask, :3] # [num_envs, 3]
            wrist_small_box_diff = wrist_pos - small_box_pos.unsqueeze(1) # [num_envs, 2, 3]
            wrist_small_box_diff = torch.flatten(wrist_small_box_diff, start_dim=1) # [num_envs, 6]
            wrist_small_box_error = torch.mean(torch.abs(wrist_small_box_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_small_box_error)
            error[mask] = wrist_small_box_error

        return reward, error, mask

    def _reward_wrist_button_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_BUTTON)

        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:, 0, :3] # [num_envs, 3], left hand, position only
            button_goal_pos = self.button_goal_pos[mask, :3] # [num_envs, 3]
            wrist_button_diff = wrist_pos - button_goal_pos # [num_envs, 3]
            wrist_button_error = torch.mean(torch.abs(wrist_button_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_button_error)
            error[mask] = wrist_button_error

        return reward, error, mask

    def _reward_right_arm_default(self):
        """
        Calculates the reward for keeping right arm joint positions close to default positions.
        """
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_BUTTON)

        if torch.any(mask):
            right_shoulder_pitch_index = 15
            joint_diff = self.dof_pos[mask] - self.default_joint_pd_target
            right_arm_diff = joint_diff[:, right_shoulder_pitch_index:] # start from right shoulder pitch
            right_arm_error = torch.mean(torch.abs(right_arm_diff), dim=1)

            reward[mask] = torch.exp(-4 * right_arm_error)
            error[mask] = right_arm_error

        return reward, error, mask

    def _reward_torso_cabinet_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_CABINET)

        if torch.any(mask):
            torso_pos = self.rigid_state[mask][:, self.torso_indices, :3].squeeze(1) # [num_envs, 3]
            cabinet_pos = self.cabinet_root_states[mask, :3] # [num_envs, 3]
            torso_cabinet_diff = cabinet_pos - torso_pos # [num_envs, 3]
            torso_cabinet_distance = torch.norm(torso_cabinet_diff, dim=1) # [num_envs]
            torso_cabinet_distance[torso_cabinet_distance < 0.1] = 0 # ignore small distance

            reward[mask] = torch.exp(-4 * torso_cabinet_distance)
            error[mask] = torso_cabinet_distance

        return reward, error, mask

    def _reward_wrist_cabinet_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_CABINET)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3] # [num_envs, 2, 3], two hands
            cabinet_pos = self.cabinet_root_states[mask, :3] # [num_envs, 3]
            wrist_cabinet_diff = wrist_pos - cabinet_pos.unsqueeze(1) # [num_envs, 2, 3]
            wrist_cabinet_diff = torch.flatten(wrist_cabinet_diff, start_dim=1) # [num_envs, 6]
            wrist_cabinet_error = torch.mean(torch.abs(wrist_cabinet_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_cabinet_error)
            error[mask] = wrist_cabinet_error

        return reward, error, mask

    def _reward_cabinet_dof_goal(self):
        """
        Calculates the reward based on the difference between the current cabinet dof positions and the target dof positions.
        """
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_CABINET)
    
        if torch.any(mask):
            cabinet_dof_diff = self.cabinet_dof_state[mask][:, :, 0] - self.cabinet_dof_goal # [num_envs, 2]
            cabinet_dof_error = torch.mean(torch.abs(cabinet_dof_diff), dim=1)

            reward[mask] = torch.exp(-4 * cabinet_dof_error)
            error[mask] = cabinet_dof_error

        return reward, error, mask
    
    def _reward_big_box_goal_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        carry_mask = (self.task_ids == self.cfg.env.TASK_CARRY)
        lift_mask = (self.task_ids == self.cfg.env.TASK_LIFT)
    
        if torch.any(carry_mask):
            big_box_pos_diff = self.big_box_root_states[carry_mask, :3] - self.big_box_goal_pos[carry_mask]
            big_box_pos_error = torch.mean(torch.abs(big_box_pos_diff), dim=1)

            reward[carry_mask] = torch.exp(-4 * big_box_pos_error)
            error[carry_mask] = big_box_pos_error

        if torch.any(lift_mask):
            big_box_pos_diff = self.big_box_root_states[lift_mask, :3] - self.big_box_goal_pos[lift_mask]
            big_box_pos_diff = big_box_pos_diff[:, 2:3] # only z axis
            big_box_pos_error = torch.mean(torch.abs(big_box_pos_diff), dim=1)

            reward[lift_mask] = torch.exp(-4 * big_box_pos_error)
            error[lift_mask] = big_box_pos_error

        return reward, error, carry_mask | lift_mask

    def _reward_wrist_big_box_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_CARRY) | (self.task_ids == self.cfg.env.TASK_LIFT)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos = wrist_pos[:,:,:3] # [num_envs, 2, 3], two hands, position only
            big_box_pos = self.big_box_root_states[mask, :3] # [num_envs, 3]
            big_box_handle_left = big_box_pos.clone()
            big_box_handle_right = big_box_pos.clone()
            # box_handle_left[:, 1] += 0.4 * self.cfg.asset.box_size[1]
            # box_handle_left[:, 2] += 0.25 * self.cfg.asset.box_size[2]
            # box_handle_right[:, 1] -= 0.4 * self.cfg.asset.box_size[1]
            # box_handle_right[:, 2] += 0.25 * self.cfg.asset.box_size[2]
            big_box_handle_pos = torch.stack([big_box_handle_left, big_box_handle_right], dim=1) # [num_envs, 2, 3]
            wrist_big_box_diff = wrist_pos - big_box_handle_pos # [num_envs, 2, 3]
            wrist_pos_diff = torch.flatten(wrist_big_box_diff, start_dim=1) # [num_envs, 6]
            wrist_big_box_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_big_box_error)
            error[mask] = wrist_big_box_error

        return reward, error, mask
    
    def _reward_wrist_ref_wrist_distance(self):
        reward = torch.zeros(self.num_envs, device=self.device)
        error = torch.zeros(self.num_envs, device=self.device)

        mask = (self.task_ids == self.cfg.env.TASK_REACH)
    
        if torch.any(mask):
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7] # [num_envs, 2, 7], two hands
            wrist_pos_diff = wrist_pos[:,:,:3] - self.ref_wrist_pos[mask][:,:,:3] # [num_envs, 2, 3], two hands, position only
            wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1) # [num_envs, 6]
            wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)

            reward[mask] = torch.exp(-4 * wrist_pos_error)
            error[mask] = wrist_pos_error

        return reward, error, mask