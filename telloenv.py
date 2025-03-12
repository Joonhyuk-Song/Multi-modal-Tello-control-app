
import gym
import numpy as np
from djitellopy import Tello
from gym import spaces
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import cv2
from commander import Controller
import time
from stable_baselines3.common.callbacks import BaseCallback

class TelloEnv(gym.Env):
    def __init__(self):
        super(TelloEnv, self).__init__()

        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.action_space = spaces.Discrete(6)

        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self):
        """Reset the environment by landing and taking off."""
        self.tello.land()
        self.tello.takeoff()

        return self._get_observation()
    
    def step(self, action):
        """Execute an action and return observation, reward, done, info."""
        self.timestep += 1  
        temp_timestep = self.timestep % 100  # Prevent oscillation issues

        move_distance = min(80, self.dist / (5 + temp_timestep / 100) + 20)
        angle = int(self.dist / 50) + 1

        if action == 0:
            self.tello.move_forward(move_distance)
        elif action == 1:
            self.tello.move_back(move_distance)
        elif action == 2:
            self.tello.move_up(move_distance)
        elif action == 3:  # Moving down with checks
            step_size = move_distance / 5
            for _ in range(5):
                if self.tello.get_height() < 50:
                    break
                self.tello.move_down(step_size)
        elif action == 4:
            self.tello.rotate_clockwise(angle)
        elif action == 5:
            self.tello.rotate_counter_clockwise(angle)

        # Get new observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(obs)

        # Define termination condition
        done = self.timestep >= self.max_steps  # Stop episode after max_steps

        return obs, reward, done, {}



    def _get_observation(self,env:Controller):
        """Capture frame and detect bounding box."""
        frame = self.tello.get_frame_read().frame
        frame = cv2.resize(frame, (640, 640))
        xmin, ymin, xmax, ymax = env.object_detection_with_yolo(frame)  #from commader.py 

        return np.array([xmin/640, ymin/640, xmax/640, ymax/640], dtype=np.float32)

    def _compute_reward(self, obs):
        """Reward based on bbox center proximity to image center and bbox size."""
        xmin, ymin, xmax, ymax = obs * [640, 640, 640, 640]
        bbox_center_x = (xmin + xmax) / 2
        bbox_center_y = (ymin + ymax) / 2

        # Image center
        img_center_x, img_center_y = 640 / 2, 640 / 2

        # Compute distance from center
        self.dist = np.sqrt((bbox_center_x - img_center_x) ** 2 + (bbox_center_y - img_center_y) ** 2)

        # Compute bounding box area
        bbox_area = (xmax - xmin) * (ymax - ymin)

        max_bbox_area = 200 * 200
        area_reward = bbox_area / 10000
        if bbox_area > max_bbox_area:
            area_reward=-(bbox_area-max_bbox_area)/10000+max_bbox_area/10000

        reward = -self.dist/ 100 +area_reward
    def render(self, mode="human"):
        frame = self.tello.get_frame_read().frame
        cv2.imshow("Tello Camera Feed", frame)
        cv2.waitKey(1)  # Display the frame for 1 millisecond


    def close(self):
        self.tello.land()
        self.tello.streamoff()
        self.tello.end()



class TimeLimitCallback(BaseCallback):
    def __init__(self, save_interval=600, save_path="ppo_tello"):
        super(TimeLimitCallback, self).__init__()
        self.save_interval = save_interval
        self.save_path = save_path
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()  # Record the start time
        return super()._on_training_start()

    def _on_step(self) -> bool:
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.save_interval:
            self.model.save(self.save_path)
            
            # Land the drone
            self.env.tello.land()
            return False  # End training after saving model and landing
        return True