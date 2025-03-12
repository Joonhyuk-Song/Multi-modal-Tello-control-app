from transformers import pipeline
from ultralytics import YOLO
import cv2
import torch
import lap
from cv2 import imshow
import time
from stable_baselines3 import PPO
from telloenv import TelloEnv, TimeLimitCallback
import ast
import threading

class Controller:
    def __init__(self, model_name="google/flan-t5-large", object_detection_model="yolov8n.pt"):
        self.qa_pipeline = pipeline("text2text-generation", model=model_name)
        self.model1 = YOLO(object_detection_model)
        self.context = context = """Convert the input into a Python list.
            The first element must be the action (one of ['front', 'back', 'down', 'up', 'left', 'right', 'detect']).
            The second element are two cases. If the first element is not 'detect' the second element will be a number if the action is 'detect the second element will be a string. The list has several list in it.

            Examples:
            Input: I want to move forward 100
            Output: ["front", 100]

            Input: I wanna move upward 100
            Output: ["up", 100]

            Input: Drone move right 200 and go up 50 and search for the dog
            Output: [["right", 200], ["up", 50], ['detect', 'dog']]

            Input: drone move right 200 and go up 50 and go left 100. after that detect a car
            Output: [['right', '200'], ['up', '50'], ['left', 100], ['detect', 'car']]

            Now, generate an output from the input in the same format.
            """
    
    def generate_output(self, question):
        answer = self.qa_pipeline(f"{self.context} Input: {question}", max_length=30)

        list_str = answer[0]["generated_text"]
        list_str = list_str.replace("Output:", "").strip()

        output = ast.literal_eval(list_str)
 
        return output     #this is a list

    def object_detection_with_yolo(self, image):
        results = self.model1.track(image, persist=True)  # Enable tracking persistence
  
        # Process results to find object and track it
        for r in results:
            if r.boxes is not None and hasattr(r.boxes, "id"):
                for box, track_id, class_id in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls):
                    if self.model1.names[int(class_id)] == self.object_to_detect and track_id == 1:
                        x1, y1, x2, y2 = map(int, box.tolist())  # Convert to integer coordinates
        return x1, y1, x2, y2


    def inference_with_PPO(self, env:TelloEnv):
        def run_inference():
            save_path = r"C:\Users\joons\OneDrive\Desktop\PES_Project\venv1\model\ppo_tello"
            model = PPO.load(save_path)
            obs = env.reset()
            self.state = True
            while self.state:
                action, _states = model.predict(obs)
                obs, reward, done, _ = env.step(action)
            env.close()

        thread = threading.Thread(target=run_inference, daemon=True)
        thread.start()


    def matcher(self, input_list, env: TelloEnv):
        self.command_list.remove(input_list)  #releasing the selected command
        action = input_list[0]
        self.state=False
        
        if action in ['front', 'back', 'down', 'up', 'left', 'right']:
            distance = float(input_list[1])

        match action:
            case 'front':
                env.tello.move_forward(distance)
                
            
            case 'back':
                env.tello.move_back(distance)
            
            case 'down':
                env.tello.move_down(distance)
            
            case 'up':
                env.tello.move_up(distance)

            case 'left':
                env.tello.rotate_counter_clockwise(distance)
            case 'right':
                env.tello.rotate_clockwise(distance)
            
            case 'detect':
                self.object_to_detect = input_list[1]
                self.inference_with_PPO()


    def commendor(self, command):
        self.command_list=self.generate_output(command)
        for item in self.command_list[:]:
            self.command_list.remove(item)
            self.matcher(item)


