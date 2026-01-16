import isaacgym
from legged_gym.utils.task_registry import task_registry
import torch

def test_env():
    # 1. Tạo môi trường
    env, env_cfg = task_registry.make_env(name="h1_unified", args=None)
    
    # 2. Reset lần đầu
    env.reset()
    
    # 3. Vòng lặp mô phỏng
    print("Đang chạy mô phỏng... Nhấn Esc trên cửa sổ GUI để thoát.")
    while True:
        # Robot H1 có khoảng 19 joint
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        
        # Step môi trường
        obs, privileged_obs, rewards, dones, infos = env.step(actions)

if __name__ == '__main__':
    # Giả lập arguments nế cần thiết
    class Args:
        task = "h1_unified"
        num_envs = 10 # Chỉ tạo 10 môi trường để load cho nhẹ máy tính
        headless = False # Hiện GUI
        physics_engine = "physx"
        sim_device = "cuda:0"
        rl_device = "cuda:0"
        test = True
        checkpoint = None
    
    test_env()