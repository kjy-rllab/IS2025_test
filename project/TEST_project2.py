
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

from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,RationalQuadratic, WhiteKernel as WK
from joblib import dump, load
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import random
import time

###################################################
########## YOU MUST CHANGE THIS PART ##############

TEAM_NAME = "TEST"

###################################################
###################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help="Set seed number.")
    parser.add_argument("--env_config", default="configs/env.yaml", type=str, help="Path to environment config file (.yaml)") # or ../..
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str, help="Path to dynamic config file (.yaml)") # or ../..
    parser.add_argument("--render", default=True, action='store_true', help="Whether to render or not.")
    parser.add_argument("--no_render", default=False, action='store_true', help="No rendering.")
    parser.add_argument("--mode", default='val', type=str, help="Whether train new model or not")
    parser.add_argument("--traj_dir", default="trajectory", type=str, help="Saved trajectory path relative to 'IS_TEAMNAME/project/'")
    parser.add_argument("--model_dir", default="model", type=str, help="Model path relative to 'IS_TEAMNAME/project/'")
    
    ###################################################
    ########## YOU CAN ONLY CHANGE THIS PART ##########
    """
    Change the name as you want.
    Note that this will used for evaluation by server as well.
    """
    parser.add_argument("--model_name", default="gp_60_super_light3.pkl", type=str, help="model name to save and use")
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


class GaussianProcess(Node):
    def __init__(self, args):
        super().__init__(f"{TEAM_NAME}_project1")
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
        
        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        1) Choose the maps to use for training as expert demonstration.
        2) Define your model and other configurations for pre/post-processing.
        """
        self.train_maps = ['map1', 'map2', 'map3', 'map4', 'map5', 'map6', 'map7', 'map8', 'map9', 'map10', 'map11', 'map12', 'map13', 'map14', 'map15', 'map16', 'map17', 'map18', 'map19' ]
        self.kernel = (
        C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
         
          + WK(1e-4, (1e-4,1)))
        #+ RationalQuadratic(length_scale= 1.0, alpha= 0.5, length_scale_bounds=(1e-5, 1e3))
        self.alpha = 1e-7
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha, n_restarts_optimizer=4)
        self.pca = PCA(n_components = 120)
        
        ###################################################
        ###################################################
        

        self.load()
        self.get_logger().info(">>> Running Project 2 for TEAM {}".format(TEAM_NAME))
        
    def train(self):
        self.get_logger().info(">>> Start model training")
        
        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        1) load your expert demonstration.
        2) Fit GP model, which gets lidar observation as input, and gives action as output.
           We recommend to pre/post-process observations and actions for better performance (e.g. normalization).
        """
        
        #random.shuffle(self.train_maps)

        train_obs_list = np.empty((0, 720))
        train_act_list = np.empty((0,1))

        for map in self.train_maps:
            obs_list = load(os.path.join(self.traj_dir, f"obs_list_{map}.pkl"))
            act_list = load(os.path.join(self.traj_dir, f"act_list_{map}.pkl"))
            act_list = act_list[:,0]

            #print(obs_list.shape)
            #print(act_list.shape)
            save_step = 4
            curve_step = 1
            prev_act = None
            delta = 0
            
            for i in range(act_list.shape[0]):
                if(i%save_step == 0):
                    train_obs_list = np.concatenate((train_obs_list, obs_list[i].reshape(1,-1)), axis = 0)
                    train_act_list = np.concatenate((train_act_list, act_list[i].reshape(1,-1)),axis = 0)
                else:
                    if(prev_act is not None):
                        delta = act_list[i] - prev_act
                        if(abs(delta) > 0.02 and i%curve_step==0):
                            train_obs_list = np.concatenate((train_obs_list, obs_list[i].reshape(1,-1)), axis = 0)
                            train_act_list = np.concatenate((train_act_list, act_list[i].reshape(1,-1)),axis = 0)
                prev_act = act_list[i]
            #print(f"{map} loading complete!")
        #print("All maps are loaded! Starting Histogram Filtering")
        #dump([train_obs_list, train_act_list], os.path.join(self.traj_dir, "flitered_data"))
        #[train_obs_list, train_act_list] = load(os.path.join(self.traj_dir, "flitered_data"))

        hist, bin_edges = np.histogram(train_act_list.flatten(), bins=15)

        weights = 1.0 / (hist + 1e-6)  # Inverse frequency weights to balance distribution
        bin_indices = np.digitize(train_act_list.flatten(), bins=bin_edges[:-1]) - 1
        probabilities = weights[bin_indices]
        probabilities /= probabilities.sum()  # Normalize probabilities

        # Perform weighted sampling
        sampled_indices = np.random.choice(len(train_act_list), size=len(train_act_list) // 5, p=probabilities)
        train_obs_list = train_obs_list[sampled_indices]
        train_act_list = train_act_list[sampled_indices]
        #print("histogram filtering finished")

        # 중복 제거
        unique_obs, unique_indices = np.unique(train_obs_list, axis=0, return_index=True)
        train_obs_list = unique_obs
        train_act_list = train_act_list[unique_indices]

        train_obs_list = normalize(train_obs_list, norm='l2')
        train_obs_list = self.pca.fit_transform(train_obs_list)
        train_obs_list, train_act_list = shuffle(train_obs_list, train_act_list, random_state = 18)
        #print(train_obs_list.shape)
        #start_time = time.time()
        self.model.fit(train_obs_list, train_act_list)
        #end_time = time.time()

        #self.get_logger().info(f"Model training completed in {end_time - start_time} seconds.")


        ###################################################
        ###################################################

        os.makedirs(self.model_dir, exist_ok=True)

        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        Save the file containing trained model and configuration for pre/post-processing.
        """
        model_dict = {'gpr': self.model, 'pca': self.pca}
        #model_dict = self.model
        dump(model_dict, os.path.join(self.model_dir, f"{self.model_name}"))
        ###################################################
        ###################################################
        
        #self.get_logger().info(">>> Trained model {} is saved".format(self.model_name))
        
            
    def load(self):
        if self.mode == 'val':
            assert os.path.exists(self.model_path)
            ###################################################
            ########## YOU CAN ONLY CHANGE THIS PART ##########
            """
            Load the trained model and configurations for pre/post-processing.
            """
            #self.model = load(self.model_path)
            #self.model = self.model.set_params(**model_params)
            model_dict = load(self.model_path)
            self.model = model_dict['gpr']
            self.pca = model_dict['pca']
            ###################################################
            ###################################################
        elif self.mode == 'train':
            self.train()
        else:
            raise AssertionError("mode should be one of 'train' or 'val'.")   

    def get_action(self, obs):
        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """
        1) Pre-process the observation input, which is current 'scan' data.
        2) Get predicted action from the model.
        3) Post-process the action. Be sure to satisfy the limitation of steer and speed values.
        """
        # Pre-processing // obs shape: (720,) -> (1, 720)
        obs = np.array(obs)
        obs = normalize(obs.reshape(1, -1), axis = 1)
        obs = self.pca.transform(obs)

        # Prediction
        steer = self.model.predict(obs)
        speed = self.max_speed
        action = np.array([steer[0], speed])
        action = action.reshape(1, -1)

        ###################################################
        ###################################################
        return action

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
    node = GaussianProcess(args)
    rclpy.spin(node)


if __name__ == '__main__':
    main()