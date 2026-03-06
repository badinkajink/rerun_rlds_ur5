import tensorflow_datasets as tfds
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

DATA_DIR = "data/deligrasp_dataset"
REPO_ID = "local/deligrasp_dataset_LeRobot"
FPS = 10

def smol():
    features = {
        "observation.images.image": {
            "dtype": "image",
            "shape": (3, 480, 640), 
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist_image": {
            "dtype": "image",
            "shape": (3, 480, 640), 
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),         
            "names": [
                "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", 
                "tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw", 
                "tcp_vx", "tcp_vy", "tcp_vz", "gripper"
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (9,),          
            "names": [
                "dx", "dy", "dz", "droll", "dpitch", "dyaw", 
                "grip_action_1", "grip_action_2", "grip_action_3"
            ],
        }
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
    )

    builder = tfds.builder_from_directory(DATA_DIR)
    raw_dataset = builder.as_dataset(split="train")

    for episode_idx, episode in enumerate(raw_dataset):
        print(f"Processing episode {episode_idx}...")
        
        for step_idx, step in enumerate(episode["steps"].as_numpy_iterator()):
            
            img = step["observation"]["image"]
            img_chw = np.transpose(img, (2, 0, 1)) 

            wimg = step["observation"]["wrist_image"]
            wimg_chw = np.transpose(wimg, (2, 0, 1)) 
            
            dataset.add_frame({
                "observation.images.image": img_chw,
                "observation.images.wrist_image": wimg_chw, 
                "observation.state": step["observation"]["state"].astype(np.float32),
                "action": step["action"].astype(np.float32),
                "task": step["subtask"].decode("utf-8")
            })
            
        dataset.save_episode()

    if hasattr(dataset, "finalize"):
        dataset.finalize()      
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()   

    print("Dataset successfully converted to LeRobot format!(In a format friendly to SmolVLA)")


def standard():  
    features = {
        "image": {
            "dtype": "image",
            "shape": (3, 480, 640), 
            "names": ["channels", "height", "width"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (3, 480, 640), 
            "names": ["channels", "height", "width"],
        },

        "state": {
            "dtype": "float32",
            "shape": (16,),         
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (9,),          
            "names": ["action"],
        }
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
    )

    builder = tfds.builder_from_directory(DATA_DIR)
    raw_dataset = builder.as_dataset(split="train")

    for episode_idx, episode in enumerate(raw_dataset):
        print(f"Processing episode {episode_idx}...")
        
        for step_idx, step in enumerate(episode["steps"].as_numpy_iterator()):
            
            img = step["observation"]["image"]
            img_chw = np.transpose(img, (2, 0, 1)) 

            wimg = step["observation"]["wrist_image"]
            wimg_chw = np.transpose(wimg, (2, 0, 1)) 
            
            # LeRobot handles all timestamps and indices natively.
            dataset.add_frame({
                "image": img_chw,
                "wrist_image": wimg_chw, 
                "state": step["observation"]["state"].astype(np.float32),
                "action": step["action"].astype(np.float32),
                "task": step["subtask"].decode("utf-8")
            })
            
        dataset.save_episode()

    if hasattr(dataset, "finalize"):
        dataset.finalize()      
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()   

    print("Dataset successfully converted to LeRobot format!")