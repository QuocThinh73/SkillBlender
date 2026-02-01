from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1MultitaskCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # Observation
        num_actions = 19
        
        num_envs = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1_wrist.urdf"

        terminate_after_contacts_on = []

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

        measure_heights = False
        
    class init_state(LeggedRobotCfg.init_state):
        pass

    class control(LeggedRobotCfg.control):
        pass

    class sim(LeggedRobotCfg.sim):
        pass

    class commands(LeggedRobotCfg.commands):
        

        class ranges:
            lin_vel_x = [-0, 0]     # min max [m/s]
            lin_vel_y = [-0, 0]     # min max [m/s]
            ang_vel_yaw = [-0, 0]   # min max [rad/s]
            heading = [-0, 0]

    class rewards:
        only_positive_rewards = True

        class scales:
            pass

    class sensor(LeggedRobotCfg.sensor):
        pass

class H1MultitaskCfgPPO(LeggedRobotCfgPPO):
    pass