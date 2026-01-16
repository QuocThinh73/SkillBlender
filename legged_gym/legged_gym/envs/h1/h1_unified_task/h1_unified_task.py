import torch
import random
import numpy as np
import math

from legged_gym.envs.legged_robot import LeggedRobot
from isaacgym import gymtorch, gymapi


class H1UnifiedTask(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        self.ball_idxs = []
        
    def _create_envs(self):
        super()._create_envs()
        
        # Load Assets
        ## Task Ball
        ### Door
        door_assets = []
        door_pose = gymapi.Transform()
        for dim in self.cfg.asset.door_dims:
            door_dims = gymapi.Vec3(*dim)
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            door_asset = self.gym.create_box(self.sim, door_dims.x, door_dims.y, door_dims.z, asset_options)
            door_assets.append(door_asset)
        
        ### Ball
        ball_size = self.cfg.asset.ball_size
        asset_options = gymapi.AssetOptions()
        ball_asset = self.gym.create_sphere(self.sim, ball_size, asset_options)
        ball_pose = gymapi.Transform()
        self.ball_idxs = []
        
        ## Task Box
            
        ## Task Button
        
        ## Task Cabinet
        
        ## Task Carry
        
        ## Task Lift
        
        ## Task Reach
        
        ## Task Transfer
        
        # Create Actors
        for i in range(self.num_envs):
            env_handle = self.envs[i]
            pos = self.env_origins[i].clone()
            ## Task Ball
            ### Add Door
            for door_i, door_asset in enumerate(door_assets):
                door_pose.p = gymapi.Vec3(*(pos[:3] + torch.tensor(self.cfg.asset.door_offsets[door_i], device=self.device)))
                door_handle = self.gym.create_actor(env_handle, door_asset, door_pose, f"door_{door_i}", i, 0)
            ### Add Ball
            ball_pose.p.x = np.random.uniform(*self.cfg.asset.ball_range_x)
            ball_pose.p.y = np.random.uniform(*self.cfg.asset.ball_range_y)
            ball_pose.p.z = 0.5 * ball_size
            ball_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "ball", i, 0)
            ### Change ball actor properties
            ball_rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            for prop in ball_rigid_body_props:
                prop.mass = random.uniform(*self.cfg.asset.ball_range_mass) # change mass here!
     
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.ball_idxs.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))

            ## Task Box
            
            ## Task Button
            
            ## Task Cabinet
            
            ## Task Carry
            
            ## Task Lift
            
            ## Task Reach
            
            ## Task Transfer

        self.ball_idxs = torch.tensor(self.ball_idxs, device=self.device)
        
    def _init_buffers(self):
        super()._init_buffers()
        
        # Task Ball
        self.ball_root_states = self.root_states[self.ball_idxs]
        
        # Task Box
        
        # Task Button
        
        # Task Cabinet
        
        # Task Carry
        
        # Task Lift
        
        # Task Reach
        
        # Task Transfer