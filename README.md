# Rerun visualizations with RLDS data on a UR5 robot
Python package for reading RLDS format data into the [Rerun](https://rerun.io/) data viewer, adapted from the [DROID equivalent](https://github.com/rerun-io/python-example-droid-dataset).

Built for [Just Add Force](justaddforce.github.io) data, downloadable from here: https://huggingface.co/datasets/correlllab/justaddforce-data

## Setup

**Prerequisite:** You must have [Conda](https://docs.conda.io/en/latest/) installed on your machine to use the setup scripts.

1. Clone and `cd` into this repository.
2. Run the appropriate setup script for your operating system to automatically create the environment and install the required dependencies:
   * **Windows:** Run `setup_env.bat`
   * **Mac/Linux:** Run `./setup_env.sh` (you may need to run `chmod +x setup_env.sh` first)
3. Once the setup script finishes, activate the new environment:
   `conda activate deligrasp_env`
4. Install this package in editable mode:
   `pip install -e .`
5. Then, install all the required libraries:
   `pip install -r requirements.txt`
6. Finally, run the python script to get your data up-to-date
   `python download_deligrasp_data.py`

## Usage

To pop up a viewer of the full RLDS data, run:
`python -m rerun_rlds_ur5.rlds --data \justaddforce_dataset\deligrasp_dataset\1.0.0 --type rlds`

To view a single sample trajectory, extract the path to the provided test file and run:
`python -m rerun_rlds_ur5.rlds --data tests/episode_yellow_rubber_duck.npy --type deligrasp`

**Visualizing LeRobot Data:**
You can now also visualize the data in LeRobot format by running:
`rerun justaddforce_dataset\deligrasp_dataset_Lerobot`

You should hopefully see something resembling [this](https://bsky.app/profile/wxie.bsky.social/post/3ljb5id5lms2m).