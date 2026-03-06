import json
import numpy as np
import pandas as pd
import glob
import os

def generate_lerobot_stats(data_directory):
    # Find all parquet files in your data chunks
    parquet_files = glob.glob(f"{data_directory}/**/*.parquet", recursive=True)
    
    if not parquet_files:
        print("No parquet files found. Check your data directory.")
        return

    all_actions = []
    all_states = []

    print(f"Processing {len(parquet_files)} files...")
    
    # Extract data from parquet files
    for file in parquet_files:
        df = pd.read_parquet(file)
        # The data in parquet is usually stored as arrays/lists per row
        all_actions.extend(np.stack(df['action'].values))
        
        # FIX: Updated column name to match the new Hugging Face format
        all_states.extend(np.stack(df['observation.state'].values)) 

    # Convert to numpy arrays for fast stats computation
    actions_np = np.array(all_actions, dtype=np.float32)
    states_np = np.array(all_states, dtype=np.float32)

    # Compute min, max, mean, and std (must be converted back to lists for JSON)
    stats = {
        "action": {
            "min": actions_np.min(axis=0).tolist(),
            "max": actions_np.max(axis=0).tolist(),
            "mean": actions_np.mean(axis=0).tolist(),
            "std": actions_np.std(axis=0).tolist()
        },
        "observation.state": {
            "min": states_np.min(axis=0).tolist(),
            "max": states_np.max(axis=0).tolist(),
            "mean": states_np.mean(axis=0).tolist(),
            "std": states_np.std(axis=0).tolist()
        }
    }

    # Ensure meta directory exists
    os.makedirs("meta", exist_ok=True)
    
    # Save to stats.json
    with open("path_to_your_data/meta/stats.json", "w") as f:
        json.dump(stats, f, indent=4)
        
    print("Successfully generated meta/stats.json!")

if __name__ == "__main__":
    generate_lerobot_stats("path_to_your_data")  # Update this path to your actual data directory