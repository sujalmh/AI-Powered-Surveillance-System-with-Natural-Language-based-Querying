# app/detection/detect.py
import os
import threading
import subprocess
import uuid
import shutil



def start_analysis(video_path, camera_id):
    def run_analysis():
        
        save_path = f"./processed_videos/{camera_id}.mp4"
        print("save path is : ",save_path)
        command = f"python ./app/detection/working_yolov9.py -v {video_path} --savepath {save_path} --camera_id {camera_id}"
        subprocess.run(command, shell=True)
        original_save_path = f"./processed_original/{camera_id}.mp4"
        shutil.copy(video_path, original_save_path)

        print("Original video saved to:", original_save_path)
    
    thread = threading.Thread(target=run_analysis)
    thread.start()
    return "Success"


# Example usage
# vp = "C:\\Users\\Hp\\Downloads\\random.mp4"
# start_analysis(vp)
# print("Analysis started in a separate thread.")

