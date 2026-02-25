import tensorflow_datasets as tfds
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 1. Configuration
DATA_DIR = "path/to/deligrasp_dataset" 
REPO_ID = "local/deligrasp_dataset_LeRobot"
FPS = 10  

features = {
    "image": {
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
        
        # LeRobot handles all timestamps and indices natively.
        dataset.add_frame({
            "image": img_chw,
            "state": step["observation"]["state"].astype(np.float32),
            "action": step["action"].astype(np.float32),
            "task": "grasp object", 
        })
        
    dataset.save_episode()

if hasattr(dataset, "finalize"):
    dataset.finalize()      
elif hasattr(dataset, "consolidate"):
    dataset.consolidate()   

print("Dataset successfully converted to LeRobot format!")