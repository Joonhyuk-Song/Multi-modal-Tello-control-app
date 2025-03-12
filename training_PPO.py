from telloenv import TelloEnv, TimeLimitCallback
from stable_baselines3 import PPO
import os

save_dir = r"C:\Users\joons\OneDrive\Desktop\PES_Project\venv1\model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "ppo_tello")

env = TelloEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, callback=TimeLimitCallback(save_interval=600))  
model.save(save_path)
env.tello.land()
