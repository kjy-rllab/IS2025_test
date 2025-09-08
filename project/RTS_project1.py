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
from joblib import dump


###################################################
########## YOU MUST CHANGE THIS PART ##############

TEAM_NAME = "RTS"

###################################################
###################################################


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help="Set seed number.")
    parser.add_argument("--env_config", default="configs/env.yaml", type=str, help="Path to environment config file (.yaml)") # or ../..
    parser.add_argument("--dynamic_config", default="configs/dynamic.yaml", type=str, help="Path to dynamic config file (.yaml)") # or ../..
    parser.add_argument("--render", default=True, action='store_true', help="Whether to render or not.")
    parser.add_argument("--no_render", default=False, action='store_true', help="No rendering.")

    parser.add_argument("--save", default=False, action='store_true', help="Whether save trajectory or not")
    parser.add_argument("--traj_dir", default="trajectory", type=str, help="Saved trajectory path relative to 'IS_TEAMNAME/project/'")

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

    return args


class PurePursuit(Node):
    def __init__(self, args):
        super().__init__(f"{TEAM_NAME}_project1")
        self.args = args
        self.query_sub = self.create_subscription(Query, "/query", self.query_callback, 10)
        self.result_pub = self.create_publisher(Result, "/result", 10)

        self.dt = args.timestep
        self.max_speed = args.max_speed
        self.min_speed = args.min_speed
        self.max_steer = args.max_steer
        self.maps = args.maps 
        self.render = args.render
        self.time_limit = 180.0
        
        self.lookahead = 10

        self.save = args.save
        self.traj_dir = args.traj_dir
        
        ###################################################
        ########## YOU CAN ONLY CHANGE THIS PART ##########
        """ 
            Setting hyperparameter
            Recommend tuning PID coefficient P->D->I order.
            Also, Recommend set Ki extremely low.
        """
        self.Kp = 0.95
        self.Ki = 0.00005
        self.Kd = 0.05
        ###################################################
        ###################################################
        self.get_logger().info(">>> Running Project 1 for TEAM {}".format(TEAM_NAME))

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
            
            waypoints = np.stack([track.centerline.xs, track.centerline.ys]).T 
            N = waypoints.shape[0]
            
            obs, _ = env.reset(seed=self.args.seed)
            pos, yaw, scan = obs
            
            prev_err = None
            curr_err = None
            sum_err = 0.0

            step = 0
            terminate = False

            ###################################################
            ################## For Project2 ###################
            """
            Freely define data structure to save trajectories.
            You have to save 'scan' and 'action(steer, speed)' information.
            Followings are some examples.
            """
            obs_list = []
            act_list = []
            ###################################################
            ###################################################

            while True:  
                ###################################################
                ########## YOU CAN ONLY CHANGE THIS PART ##########
                """ 
                1) Find nearest waypoint from a rccar
                2) Calculate error between the rccar heading and the direction vector between the lookahead waypoint and the rccar
                3) Determine input steering of the rccar using PID controller
                4) Calculate input velocity of the rccar appropriately in terms of input steering
                """
                distance = np.linalg.norm(waypoints - pos, ord=2, axis=1)

                nearest_wp = np.argmin(distance)
                idx = min(nearest_wp + self.lookahead,N-1)
                dest_vec = waypoints[idx,:]
                direction_vec = dest_vec - waypoints[nearest_wp, :]
                
                #error calculation
                dest_yaw = np.arctan2(direction_vec[1], direction_vec[0])
                if(abs(dest_yaw-yaw)>np.pi and dest_yaw-yaw < 0):
                        dest_yaw = 2*np.pi+dest_yaw
                elif(abs(dest_yaw-yaw)>np.pi and dest_yaw-yaw > 0):
                        dest_yaw = dest_yaw - 2*np.pi

                #print(f"f{dest_yaw}, {yaw}")
                curr_err = dest_yaw - yaw
                if prev_err == None:
                    prev_err = curr_err
                steer = self.Kp * curr_err + self.Ki * self.dt * sum_err + self.Kd * (curr_err - prev_err) / self.dt
                if steer > self.max_steer:
                    steer = self.max_steer
                if steer < -self.max_steer:
                    steer = -self.max_steer
                if abs(steer) > 0.05 :
                    reduction_ratio = 0
                    speed = (self.max_steer-reduction_ratio*abs(steer))/(self.max_steer)*self.max_speed
                else:
                    speed = self.max_speed
                prev_err = curr_err
                sum_err += curr_err

                obs_list.append(scan)
                act_list.append((steer, speed))


                ###################################################
                ###################################################
                obs, _, terminate, _, info = env.step(np.array([steer, speed]))
                pos, yaw, scan = obs
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
                    if self.save:
                        os.makedirs(self.traj_dir, exist_ok=True)
                        ###################################################
                        ################## For Project2 ###################
                        """
                        Save trajectory when terminated at 'args.traj_dir'.
                        """
                        obs_list = np.array(obs_list)
                        act_list = np.array(act_list)
                        dump(obs_list, os.path.join(self.traj_dir, f"obs_list_{map}.pkl"))
                        dump(act_list, os.path.join(self.traj_dir, f"act_list_{map}.pkl"))

                        ###################################################
                        ###################################################
                        self.get_logger().info(">>> map {} trajectory saved".format(map))
                    
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
                        #elapsed_time = END_TIME - START_TIME
                        self.get_logger().info(">>> Success: {}".format(map))
                        #self.get_logger().info(f">>> Success: {elapsed_time}")
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
    node = PurePursuit(args)
    rclpy.spin(node)


if __name__ == '__main__':
    main()

