import os
import time
import argparse
import numpy as np
from ruamel.yaml import YAML
from easydict import EasyDict

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix

from message.msg import Result, Query
from rccar_gym.env_wrapper import RCCarWrapper

from joblib import load
import torch

from tqdm import tqdm
from .PPO import *
from sklearn.preprocessing import normalize

###################################################
########## YOU CAN ONLY CHANGE THIS PART  #########

"""
Freely import modules, define methods and classes, etc.
You may add other python codes, but make sure to push it to github.
To use particular modules, please let TA know to install them on the evaluation server, too.
If you want to use a deep-learning library, please use pytorch.
"""

TEAM_NAME = "RTS"

###################################################
###################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int, help="Set seed number.")
    parser.add_argument("--env_config", default="configs/env.yaml", type=str, help="Path to environment config file (.yaml)") # or ../..
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str, help="Path to dynamic config file (.yaml)") # or ../..
    parser.add_argument("--render", default=False, action='store_true', help="Whether to render or not.")
    parser.add_argument("--no_render", default=False, action='store_true', help="No rendering.")
    parser.add_argument("--mode", default='val', type=str, help="Whether train new model or not")
    parser.add_argument("--traj_dir", default="trajectory", type=str, help="Saved trajectory path relative to 'IS_TEAMNAME/project/'")
    parser.add_argument("--model_dir", default="model", type=str, help="Model path relative to 'IS_TEAMNAME/project/'")

    
    ###################################################
    ########## YOU CAN ONLY CHANGE THIS PART ##########
    """
    Change the model name as you want.
    Note that this will used for evaluation by the server as well.
    You can add any arguments you want.
    """
    parser.add_argument("--max_timesteps", default=1e4, type=int)
    parser.add_argument("--model_name", default="model3.pkl", type=str, help="model name to save and use")
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--std_coeff", default=50, type=int)
    parser.add_argument("--std_update", default=120000, type=int)
    parser.add_argument("--update_freq", default=500, type=int)
    ###################################################
    ###################################################
    
    args = parser.parse_args()
    args = EasyDict(vars(args))
    
    # render
    if args.no_render:
        args.render = False
    
    ws_path = os.path.join(get_package_prefix('rccar_bringup'), "../..")

    # map files
    args.maps = os.path.join(ws_path, 'maps')
    args.maps = [map for map in os.listdir(args.maps) if os.path.isdir(os.path.join(args.maps, map))]
    
    # configuration files
    args.env_config = os.path.join(ws_path, args.env_config)
    args.dynamic_config = os.path.join(ws_path, args.dynamic_config)
    
    with open(args.env_config, 'r') as f:
        task_args = EasyDict(YAML().load(f))
    with open(args.dynamic_config, 'r') as f:
        dynamic_args = EasyDict(YAML().load(f))

    args.update(task_args)
    args.update(dynamic_args)
    
    # Trajectory & Model Path
    project_path = os.path.join(ws_path, f"src/rccar_bringup/rccar_bringup/project/IS_{TEAM_NAME}/project")
    args.traj_dir = os.path.join(project_path, args.traj_dir)
    args.model_dir = os.path.join(project_path, args.model_dir)
    args.model_path = os.path.join(args.model_dir, args.model_name)

    return args


class PPOPolicy(Node):
    def __init__(self, args):
        super().__init__(f"{TEAM_NAME}_online")
        self.args = args
        self.mode = args.mode

        self.query_sub = self.create_subscription(Query, "/query", self.query_callback, 10)
        self.result_pub = self.create_publisher(Result, "/result", 10)

        self.dt = args.timestep
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer
        self.maps = args.maps 
        self.render = args.render
        self.time_limit = 180.0

        self.traj_dir = args.traj_dir
        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.batch_size = args.batch_size
        self.std_coeff = args.std_coeff / 100
        self.std_update = args.std_update
        self.update_freq = args.update_freq
        
    ###################################################
    ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        Freely change the codes to increase the performance.
        """
        #device = "cuda"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_maps = []#, 'map2']

        for i in range(60):
            self.train_maps.append(f"rand{i+1}")
        
        self.load()
        self.get_logger().info(">>> Running Project 3 for TEAM {}".format(TEAM_NAME))
        
    def train(self):
        self.get_logger().info(">>> Start model training")
        """
        Train and save your model.
        You can either use this part or explicitly train using other python codes.
        """
        save_model_freq = 250
        max_ep_len = 10000
        max_training_timesteps = 1e8

        print_freq = 1000
        log_freq = max_ep_len * 2

        steer_action_std = self.std_coeff *self.max_steer
        steer_action_std_decay_rate = 0.005 * self.max_steer
        steer_min_action_std = 0.05

        vel_action_std = 2.
        vel_action_std_decay_rate = 0.05
        vel_min_action_std = 0.05

        action_std_decay_freq = self.std_update
        render_freq = 500000

        ############ PPO hyper params ############
        update_timestep = self.update_freq
        K_epochs = 30

        eps_clip = 0.2
        gamma = 0.97

        lr_actor = 0.0005
        lr_critic = 0.001
        has_continuous_action_space = True
        state_dim = 720
        action_dim = 2

        success_reward = 100000
        way_point_reward = 0
        time_step_reward = -2
        failure_reward = -10000

        self.ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,self.max_steer,steer_action_std,vel_action_std, batch_size = self.batch_size)
        
        ##########################################

        env = RCCarWrapper(args=self.args, maps=[self.train_maps[0]], render_mode="human_fast" if self.render else None)
        track = env._env.unwrapped.track
        if self.render:
            env.unwrapped.add_render_callback(track.centerline.render_waypoints)
        
        time_step = 0
        i_episode = 0

        curr_map = 1
        env_time_step = 0
        env_switch_interval = 2

        while time_step <= max_training_timesteps:
            if(env_time_step == env_switch_interval):
                # curr_map += 1
                curr_map = np.random.randint(0, 60)
                env = RCCarWrapper(args=self.args, maps=[self.train_maps[curr_map]], render_mode="human_fast" if self.render else None)
                track = env._env.unwrapped.track
                if self.render:
                    env.unwrapped.add_render_callback(track.centerline.render_waypoints)
                env_time_step = 1
                
            else:
                env_time_step += 1
            state, _ = env.reset(seed=self.args.seed)
            print_running_reward = 0
            print_running_episodes = 0
            current_ep_reward = 0
            terminate = False
            flag = False
            prev_way_point = 0
            if (i_episode % render_freq == 0)and (i_episode != 0):
                n = np.random.randint(1, 61)
                eval_maps = ["map1","rand1", f"rand{n}"]
                for map in eval_maps:
                    env_eval = RCCarWrapper(args=self.args, maps=[map], render_mode="human_fast")
                    track = env_eval._env.unwrapped.track
                    env_eval.unwrapped.add_render_callback(track.centerline.render_waypoints)
                    state, _ = env_eval.reset(seed=self.args.seed)
                    for _ in range(0, max_ep_len):
                        norm_state = normalize(state[2].reshape(1, -1), axis = 1)
                        action = self.ppo_agent.policy_old.act_eval(norm_state)
                        #action = np.array([action.item(),3.999])
                        state, _, terminate, _, info = env_eval.step(action)
                        env_eval.render()
                        if terminate:
                            break
                i_episode += 1
                pass
                    
            else:
                prev_vel = 0
                for _ in range(0, max_ep_len):
                    norm_state = normalize(state[2].reshape(1, -1), axis = 1)
                    action = self.ppo_agent.select_action(norm_state)
                    #action = np.array([action.item(),3.999])
                    state, _, terminate, _, info = env.step(action)
                    
                    self.ppo_agent.buffer.is_terminals.append(terminate)
                    if terminate:     
                        if info['waypoint'] == 20:
                            # sucess
                            self.ppo_agent.buffer.rewards.append(success_reward)
                            current_ep_reward += success_reward
                        else:
                            #failure
                            self.ppo_agent.buffer.rewards.append(failure_reward)
                            current_ep_reward += failure_reward
                        print_avg_reward = current_ep_reward / print_running_episodes
                        print_avg_reward = round(print_avg_reward, 2)
                        if(i_episode%print_freq):
                            print("Episode : {} \t Timestep : {} \t Average Reward : {} \t Map_num : {} \t wp:{}/20".format(i_episode, time_step, print_avg_reward, curr_map, info['waypoint']))
                        i_episode += 1
                        break
                    else:
                        if(info['waypoint']!=prev_way_point):
                            self.ppo_agent.buffer.rewards.append(way_point_reward)
                            prev_way_point = info['waypoint']
                            current_ep_reward += way_point_reward
                        else:
                            self.ppo_agent.buffer.rewards.append(time_step_reward)
                            current_ep_reward += (-4 + action[1])- 0.1*(prev_vel-action[1])**2           
                    time_step += 1
                    print_running_episodes += 1
                    if time_step % update_timestep == 0:
                        self.ppo_agent.update()
                    if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                        self.ppo_agent.decay_action_std(steer_action_std_decay_rate, steer_min_action_std,vel_action_std_decay_rate, vel_min_action_std)

                    prev_vel = action[1].item()
                    #if(True):
                        #print(info.keys())
                        #break
            if(i_episode%save_model_freq==0):
                self.ppo_agent.save(self.model_path)
                print("Model Saved")
            


            
       
        
        self.get_logger().info(">>> Trained model {} is saved".format(self.model_name))
            
    def load(self):
        """
        Load your trained model.
        Make sure not to train a new model when self.mode == 'val'.
        """
        if self.mode == 'val':
            assert os.path.exists(self.model_dir)
            #self.ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,self.max_steer,action_std)
            
        elif self.mode == 'train':
            self.train()
        else:
            raise AssertionError("mode should be one of 'train' or 'val'.")   

    def get_action(self, obs):
        """
        Predict action using obs - 'scan' data.
        Be sure to satisfy the limitation of steer and speed values.
        """
        obs = normalize(obs.reshape(1, -1), axis = 1)
        action = self.ppo_agent.select_action(obs)
        return action
    
    ###################################################
    ###################################################

    def query_callback(self, query_msg):
        
        id = query_msg.id
        team = query_msg.team
        map = query_msg.map
        trial = query_msg.trial
        exit = query_msg.exit 
        
        result_msg = Result()
        
        START_TIME = time.time()
            
        try:
            if team != TEAM_NAME:
                return
            
            if map not in self.maps:
                END_TIME = time.time()
                result_msg.id = id
                result_msg.team = team
                result_msg.map = map
                result_msg.trial = trial
                result_msg.time = END_TIME - START_TIME
                result_msg.waypoint = 0
                result_msg.n_waypoints = 20
                result_msg.success = False
                result_msg.fail_type = "Invalid Track"
                self.get_logger().info(">>> Invalid Track")
                self.result_pub.publish(result_msg)
                return
            
            self.get_logger().info(f"[{team}] START TO EVALUATE! MAP NAME: {map}")
            
            ### New environment
            env = RCCarWrapper(args=self.args, maps=[map], render_mode="human_fast" if self.render else None)
            track = env._env.unwrapped.track
            if self.render:
                env.unwrapped.add_render_callback(track.centerline.render_waypoints)
            
            obs, _ = env.reset(seed=self.args.seed)
            _, _, scan = obs

            step = 0
            terminate = False

            while True:  

                act = self.get_action(scan)
                steer = np.clip(act[0][0], -self.max_steer, self.max_steer)
                speed = np.clip(act[0][1], self.min_speed, self.max_speed)
                
                obs, _, terminate, _, info = env.step(np.array([steer, speed]))
                _, _, scan = obs
                step += 1
                
                if self.render:
                    env.render()

                if time.time() - START_TIME > self.time_limit:
                    END_TIME = time.time()
                    result_msg.id = id
                    result_msg.team = team
                    result_msg.map = map
                    result_msg.trial = trial
                    result_msg.time = step * self.dt
                    result_msg.waypoint = info['waypoint']
                    result_msg.n_waypoints = 20
                    result_msg.success = False
                    result_msg.fail_type = "Time Out"
                    self.get_logger().info(">>> Time Out: {}".format(map))
                    self.result_pub.publish(result_msg)
                    env.close()
                    break

                if terminate:
                    END_TIME = time.time()
                    result_msg.id = id
                    result_msg.team = team
                    result_msg.map = map
                    result_msg.trial = trial
                    result_msg.time = step * self.dt 
                    result_msg.waypoint = info['waypoint']
                    result_msg.n_waypoints = 20
                    if info['waypoint'] == 20:
                        result_msg.success = True
                        result_msg.fail_type = "-"
                        self.get_logger().info(">>> Success: {}".format(map))
                    else:
                        result_msg.success = False
                        result_msg.fail_type = "Collision"
                        self.get_logger().info(">>> Collision: {}".format(map))
                    self.result_pub.publish(result_msg)
                    env.close()
                    break
        except:
            END_TIME = time.time()
            result_msg.id = id
            result_msg.team = team
            result_msg.map = map
            result_msg.trial = trial
            result_msg.time = END_TIME - START_TIME
            result_msg.waypoint = 0
            result_msg.n_waypoints = 20
            result_msg.success = False
            result_msg.fail_type = "Script Error"
            self.get_logger().info(">>> Script Error")
            self.result_pub.publish(result_msg)
        
        if exit:
            rclpy.shutdown()
        return

def main():
    args = get_args()
    rclpy.init()
    node = PPOPolicy(args)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
