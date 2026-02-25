import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from copy import deepcopy
import os
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict

class MultiActorCriticHierarchical(nn.Module):
    is_recurrent = False

    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        num_tasks,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        super(MultiActorCriticHierarchical, self).__init__()
        
        self.num_tasks = num_tasks
        self.num_actions = num_actions
        self.device = kwargs['device']
        self.obs_context_len = kwargs.get('obs_context_len', 1)
        
        # 1. THIẾT LẬP CÁC THÔNG SỐ DIMENSION CƠ BẢN
        self.frame_stack = kwargs.get('frame_stack', 1)
        self.c_frame_stack = kwargs.get('c_frame_stack', 1)
        self.max_command_dim = kwargs.get('command_dim', 14) 
        self.task_specific_obs_dims = kwargs.get('task_specific_obs_dims', [14]*num_tasks)
        self.task_specific_priv_obs_dims = kwargs.get('task_specific_priv_obs_dims', [42]*num_tasks)
        
        # 2. LOAD LOW-LEVEL EXPERT SKILLS
        self.args = kwargs['args']
        self.num_dofs = num_actions # 19
        self._get_low_level_policies(self.args, self.device, kwargs)
        
        # 3. THIẾT LẬP NHÓM BỘ PHẬN & OUTPUT DIMENSION
        self._setup_grouping()
        num_hl_output = self._get_high_level_output_dim()

        # Tính toán dimension phần common_obs
        num_single_common_obs = (num_actor_obs // self.frame_stack) - self.max_command_dim - self.num_tasks
        num_single_common_priv_obs = (num_critic_obs // self.c_frame_stack) - kwargs.get('max_privileged_obs_dim', 42) - self.num_tasks

        activation_fn = self.get_activation(activation)

        # 4. KHỞI TẠO HIGH-LEVEL ACTORS CHO TỪNG TASK
        self.hl_actors = nn.ModuleList()
        for i in range(num_tasks):
            hl_input_dim = (self.task_specific_obs_dims[i] + num_single_common_obs) * self.frame_stack
            self.hl_actors.append(
                self._build_mlp(hl_input_dim, num_hl_output, actor_hidden_dims, activation_fn)
            )

        # 5. KHỞI TẠO HIGH-LEVEL CRITICS CHO TỪNG TASK
        self.hl_critics = nn.ModuleList()
        for i in range(num_tasks):
            critic_input_dim = (self.task_specific_priv_obs_dims[i] + num_single_common_priv_obs) * self.c_frame_stack
            self.hl_critics.append(
                self._build_mlp(critic_input_dim, 1, critic_hidden_dims, activation_fn)
            )

        # Action noise chung
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print(f"Unified Multi-Task MoE: {num_tasks} High-Level Routers sharing {self.num_skills} Low-Level Experts.")
        print(f"Anatomical Routing: Enabled ({self.num_groups} Groups). Residual Actions: Enabled.")

    def _setup_grouping(self):
        """Chia 19 DoFs của robot H1 thành 3 nhóm bộ phận giải phẫu học"""
        self.group_legs = list(range(0, 10))   # 10 khớp chân & hông
        self.group_torso = [10]                # 1 khớp eo (torso)
        self.group_arms = list(range(11, 19))  # 8 khớp cho CẢ HAI cánh tay
        self.num_groups = 3
        
        # Thiết lập thông số cho Temperature Annealing
        self.tau_max = 1.0    # Nhiệt độ ban đầu (Khám phá mạnh)
        self.tau_min = 0.1    # Nhiệt độ cuối cùng (Gần với One-hot)
        self.tau_decay = 0.99995 # Hệ số giảm sau mỗi lần gọi update
        self.tau = self.tau_max

    def update_tau(self):
        """Hàm này sẽ được gọi 1 lần duy nhất bởi Runner sau mỗi Learning Iteration"""
        self.tau = max(self.tau_min, self.tau * self.tau_decay)

    def _get_high_level_output_dim(self):
        """Tính kích thước output cuối cùng của mạng High-level"""
        self.total_cmd_dim = sum([cfg.env.command_dim for cfg in self.env_cfg_list])
        self.routing_dim = self.num_skills * self.num_groups
        self.residual_dim = self.num_actions # 19
        return self.total_cmd_dim + self.routing_dim + self.residual_dim

    def _build_mlp(self, input_dim, output_dim, hidden_dims, activation):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*layers)

    def get_activation(self, act_name):
        if act_name == "elu": return nn.ELU()
        elif act_name == "selu": return nn.SELU()
        elif act_name == "relu": return nn.ReLU()
        elif act_name == "lrelu": return nn.LeakyReLU()
        elif act_name == "tanh": return nn.Tanh()
        return nn.ELU()

    def _get_one_policy(self, args, device, task, experiment_name, load_run, checkpoint):
        from legged_gym.utils import task_registry
        from rsl_rl.modules import ActorCritic
        from legged_gym.utils.helpers import get_load_path
        
        skill_args = deepcopy(args)
        assert task == experiment_name
        skill_args.task = task
        skill_args.experiment_name = experiment_name
        skill_args.load_run = load_run
        skill_args.checkpoint = checkpoint
        skill_env_cfg, skill_train_cfg = task_registry.get_cfgs(name=skill_args.task, load_run=skill_args.load_run, experiment_name=skill_args.experiment_name)
        
        skill_policy = ActorCritic(
            skill_env_cfg.env.num_observations,
            skill_env_cfg.env.num_privileged_obs,
            skill_env_cfg.env.num_actions,
            obs_context_len=1,
            **class_to_dict(skill_train_cfg)["policy"]
        ).to(device)
        
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', skill_train_cfg.runner.experiment_name)
        skill_resume_path = get_load_path(log_root, load_run=skill_args.load_run, checkpoint=skill_args.checkpoint)
        print(f"Loading {skill_args.task} expert policy from: {skill_resume_path}")
        
        try:
            loaded_dict = torch.load(skill_resume_path, map_location=device)
        except:
            loaded_dict = torch.load(skill_resume_path, map_location="cuda:0")
            
        skill_policy.load_state_dict(loaded_dict['model_state_dict'])
        
        # Đóng băng expert
        skill_policy.eval()
        for param in skill_policy.parameters():
            param.requires_grad = False
            
        return skill_policy.actor, skill_env_cfg, skill_train_cfg
        
    def _get_low_level_policies(self, args, device, kwargs):
        skill_dict = kwargs['skill_dict']
        self.skill_names = list(skill_dict.keys())
        self.policy_list = nn.ModuleList() # Sử dụng ModuleList để quản lý device tốt hơn
        self.env_cfg_list = []
        self.train_cfg_list = []
        self.low_high_list = []
        
        for key, value in skill_dict.items():
            policy, env_cfg, train_cfg = self._get_one_policy(args, device, key, value['experiment_name'], value['load_run'], value['checkpoint'])
            self.policy_list.append(policy)
            self.env_cfg_list.append(env_cfg)
            self.train_cfg_list.append(train_cfg)
            self.low_high_list.append(value['low_high'])
            
        self.num_skills = len(self.policy_list)

    def _replace_observations(self, observations, command, low_high=None):
        if low_high is not None:
            low, high = low_high
            command = torch.clamp(command, low, high)
        new_observations = observations.clone().reshape(observations.shape[0], self.frame_stack, -1)
        num_envs, frame_stack, num_single_obs = new_observations.shape
        state_dim = num_single_obs - self.max_command_dim - self.num_tasks
        
        new_command = command.reshape(num_envs, frame_stack, -1)
        replaced_observations = torch.zeros((num_envs, frame_stack, state_dim + new_command.shape[-1]), device=self.device)
        
        # Bỏ qua task_specific_obs và ghép state tĩnh với command mới
        replaced_observations[:, :, new_command.shape[-1]:] = new_observations[:, :, self.max_command_dim:-self.num_tasks]
        replaced_observations[:, :, :new_command.shape[-1]] = new_command
        return replaced_observations.reshape(num_envs, -1)

    def _parse_observations(self, obs_batch, task_id, is_critic=False):
        frame_stack = self.c_frame_stack if is_critic else self.frame_stack
        obs_reshaped = obs_batch.clone().reshape(obs_batch.shape[0], frame_stack, -1)
        num_envs, fs, max_single_obs = obs_reshaped.shape

        if is_critic:
            max_specific_dim = 42 
            true_specific_dim = self.task_specific_priv_obs_dims[task_id]
        else:
            max_specific_dim = self.max_command_dim
            true_specific_dim = self.task_specific_obs_dims[task_id]

        true_specific_obs = obs_reshaped[:, :, :true_specific_dim]
        common_obs = obs_reshaped[:, :, max_specific_dim:-self.num_tasks]
        
        parsed_obs = torch.cat((true_specific_obs, common_obs), dim=-1)
        return parsed_obs.reshape(num_envs, -1)

    def _actor(self, observations, task_ids):
        # 1. ĐIỀU HƯỚNG TỚI HIGH-LEVEL ACTORS
        raw_output_batch = torch.zeros(observations.shape[0], self._get_high_level_output_dim(), device=self.device)
        for i in range(self.num_tasks):
            mask = (task_ids == i)
            if mask.any():
                parsed_obs = self._parse_observations(observations[mask], task_id=i, is_critic=False)
                raw_output_batch[mask] = self.hl_actors[i](parsed_obs)

        # 2. CẮT OUTPUT THÀNH 3 NHÁNH (Commands, Routing Logits, Residuals)
        commands = raw_output_batch[:, :self.total_cmd_dim]
        routing_logits = raw_output_batch[:, self.total_cmd_dim : self.total_cmd_dim + self.routing_dim]
        residuals = raw_output_batch[:, -self.residual_dim:]

        # 3. GUMBEL-SOFTMAX ROUTING (HARD)
        routing_logits = routing_logits.reshape(-1, self.num_groups, self.num_skills)
        group_masks = torch.nn.functional.gumbel_softmax(routing_logits, tau=self.tau, hard=True, dim=-1)

        # Trải rộng mask từ 3 nhóm ra 19 khớp
        dof_masks = torch.zeros((observations.shape[0], self.num_skills, self.num_actions), device=self.device)
        for i in range(self.num_skills):
            dof_masks[:, i, self.group_legs] = group_masks[:, 0, i].unsqueeze(1)
            dof_masks[:, i, self.group_torso] = group_masks[:, 1, i].unsqueeze(1)
            dof_masks[:, i, self.group_arms] = group_masks[:, 2, i].unsqueeze(1)

        # 4. GỌI LOW-LEVEL EXPERTS VÀ TRỘN VỚI MASK
        means = []
        prev_dim = 0
        for i in range(self.num_skills):
            curr_dim = self.env_cfg_list[i].env.command_dim
            cmd_i = commands[:, prev_dim : prev_dim + curr_dim]
            input_to_expert_i = self._replace_observations(observations, cmd_i, low_high=self.low_high_list[i])
            
            expert_action = self.policy_list[i](input_to_expert_i)
            
            means.append(expert_action * dof_masks[:, i])
            prev_dim += curr_dim
            
        base_action = sum(means)

        # 5. CỘNG RESIDUAL ACTION (TINH CHỈNH TỪ HIGH-LEVEL)
        scaled_residuals = torch.tanh(residuals) * 0.1 # Biên độ bù trừ là +/- 0.1 rad
        final_action = base_action + scaled_residuals

        return {
            "actions_mean": final_action,
            "masks": dof_masks
        }

    # ========================== GIAO TIẾP VỚI PPO ========================== #
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, task_ids):
        if self.obs_context_len != 1:
            observations = observations[..., -1, :]
        actions_mean = self._actor(observations, task_ids)['actions_mean']
        self.distribution = Normal(actions_mean, actions_mean * 0. + self.std)

    def act(self, observations, task_ids, **kwargs):
        self.update_distribution(observations, task_ids)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, task_ids):
        if self.obs_context_len != 1:
            observations = observations[..., -1, :]
        return self._actor(observations, task_ids)['actions_mean']
    
    def act_inference_hrl(self, observations, task_ids):
        if self.obs_context_len != 1:
            observations = observations[..., -1, :]
        return self._actor(observations, task_ids)
    
    def evaluate(self, critic_observations, task_ids, **kwargs):
        if self.obs_context_len != 1:
            critic_observations = critic_observations[..., -1, :]
            
        values = torch.zeros(critic_observations.shape[0], 1, device=self.device)
        for i in range(self.num_tasks):
            mask = (task_ids == i)
            if mask.any():
                parsed_critic_obs_i = self._parse_observations(critic_observations[mask], task_id=i, is_critic=True)
                values[mask] = self.hl_critics[i](parsed_critic_obs_i)
        return values

    def reset(self, dones=None):
        pass