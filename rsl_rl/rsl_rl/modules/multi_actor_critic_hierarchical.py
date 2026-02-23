import torch
import torch.nn as nn
from torch.distributions import Normal

class MultiActorCriticHierarchical(nn.Module):
    is_recurrent = False # Giữ False nếu bạn dùng MLP thuần túy

    def __init__(self,  num_actor_obs, # Max obs dim từ env (để dự phòng)
                        num_critic_obs, # Max critic obs dim từ env
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
        activation_fn = self.get_activation(activation)

        # Lấy thông số từ kwargs (đã định nghĩa trong config)
        self.frame_stack = kwargs.get('frame_stack', 1)
        self.c_frame_stack = kwargs.get('c_frame_stack', 1) # Nếu critic dùng frame_stack khác
        
        self.max_command_dim = kwargs.get('command_dim', 14) 
        self.task_specific_obs_dims = kwargs.get('task_specific_obs_dims', [14]*num_tasks)
        self.task_specific_priv_obs_dims = kwargs.get('task_specific_priv_obs_dims', [42]*num_tasks)

        # Số chiều của phần common_obs (state tĩnh: q, dq, actions, v.v...)
        # Cách tính: tổng_max_obs (cho 1 frame) - max_command_dim - num_tasks(one_hot)
        num_single_common_obs = (num_actor_obs // self.frame_stack) - self.max_command_dim - self.num_tasks
        num_single_common_priv_obs = (num_critic_obs // self.c_frame_stack) - kwargs.get('max_privileged_obs_dim', 42) - self.num_tasks

        self.actors = nn.ModuleList()
        self.critics = nn.ModuleList()

        for i in range(num_tasks):
            # Tính toán input_dim chính xác cho Actor thứ i
            # = (task_specific_dim_i + common_obs_dim) * frame_stack
            actor_input_dim = (self.task_specific_obs_dims[i] + num_single_common_obs) * self.frame_stack
            self.actors.append(
                self._build_mlp(actor_input_dim, num_actions, actor_hidden_dims, activation_fn)
            )

            # Tính toán input_dim chính xác cho Critic thứ i
            critic_input_dim = (self.task_specific_priv_obs_dims[i] + num_single_common_priv_obs) * self.c_frame_stack
            self.critics.append(
                self._build_mlp(critic_input_dim, 1, critic_hidden_dims, activation_fn)
            )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print(f"Multi-Task Actor: {num_tasks} experts created with DYNAMIC input dimensions.")

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

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def _parse_observations(self, obs_batch, task_id, is_critic=False):
        """
        Hàm này nhận batch obs (đã bị đệm zeros) và cắt lấy đúng số chiều cần thiết cho task_id.
        Logic: 
        1. Tách obs ra thành [batch, frame_stack, single_obs_dim]
        2. Cắt bỏ phần task_one_hot (vì mạng policy đã được tách riêng rẽ, không cần one_hot nữa)
        3. Cắt phần max_command_dim thành true_command_dim
        4. Ghép lại
        """
        frame_stack = self.c_frame_stack if is_critic else self.frame_stack
        
        # Reshape để xử lý từng frame
        obs_reshaped = obs_batch.clone().reshape(obs_batch.shape[0], frame_stack, -1)
        num_envs, fs, max_single_obs = obs_reshaped.shape

        if is_critic:
            max_specific_dim = 42 # max_privileged_obs_dim từ env config
            true_specific_dim = self.task_specific_priv_obs_dims[task_id]
        else:
            max_specific_dim = self.max_command_dim
            true_specific_dim = self.task_specific_obs_dims[task_id]

        # Trong h1_unified_task.py, obs được ghép theo thứ tự: 
        # [task_specific_obs (max_dim), common_obs_khác, task_one_hot (8)]
        
        # Lấy phần task_specific_obs_buf (nhưng chỉ lấy số chiều thực sự cần)
        true_specific_obs = obs_reshaped[:, :, :true_specific_dim]
        
        # Lấy phần common_obs_buf (bỏ qua max_specific_dim ban đầu, và bỏ qua phần task_one_hot ở cuối cùng)
        common_obs = obs_reshaped[:, :, max_specific_dim:-self.num_tasks]

        # Ghép lại
        parsed_obs = torch.cat((true_specific_obs, common_obs), dim=-1)
        
        # Flatten lại theo frame_stack
        return parsed_obs.reshape(num_envs, -1)

    def update_distribution(self, observations, task_ids):
        actions_mean = torch.zeros(observations.shape[0], self.num_actions, device=observations.device)
        
        for i in range(self.num_tasks):
            mask = (task_ids == i)
            if mask.any():
                # Chỉ extract những env thuộc task i
                obs_task_i = observations[mask]
                
                # Gọi hàm parse để cắt gọn obs
                parsed_obs_i = self._parse_observations(obs_task_i, task_id=i, is_critic=False)
                
                # Đưa vào đúng Actor
                actions_mean[mask] = self.actors[i](parsed_obs_i)
        
        self.distribution = Normal(actions_mean, actions_mean * 0. + self.std)

    def act(self, observations, task_ids, **kwargs):
        self.update_distribution(observations, task_ids)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, task_ids):
        actions_mean = torch.zeros(observations.shape[0], self.num_actions, device=observations.device)
        for i in range(self.num_tasks):
            mask = (task_ids == i)
            if mask.any():
                actions_mean[mask] = self.actors[i](observations[mask])
        return actions_mean

    def evaluate(self, critic_observations, task_ids, **kwargs):
        values = torch.zeros(critic_observations.shape[0], 1, device=critic_observations.device)
        for i in range(self.num_tasks):
            mask = (task_ids == i)
            if mask.any():
                # Lấy obs của task i
                critic_obs_task_i = critic_observations[mask]
                
                # Parse critic obs
                parsed_critic_obs_i = self._parse_observations(critic_obs_task_i, task_id=i, is_critic=True)
                
                # Đưa vào Critic
                values[mask] = self.critics[i](parsed_critic_obs_i)
        return values

    def reset(self, dones=None):
        pass