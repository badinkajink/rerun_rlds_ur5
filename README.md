# Rerun visualizations with RLDS data on a UR5 robot
Python package for reading RLDS format data into the [Rerun](https://rerun.io/) data viewer, adapted from the [DROID equivalent](https://github.com/rerun-io/python-example-droid-dataset).

Built for [Just Add Force](justaddforce.github.io) data, downloadable from here: https://huggingface.co/datasets/correlllab/justaddforce-data

Extract the `deligrasp_dataset` or `deligrasp_dataset_grasponly` folders to your top-level `tensorflow_datasets` directory. By default `deligrasp_dataset` is loaded.

Then, clone and cd into this repo and run `pip install -e .`

Finally, run `python -m rerun_rlds_ur5.rlds` to pop up a viewer of the data.

To view a single sample trajectory, extract the path to the provided `tests/episode_yellow_rubber_duck.npy` and run `python -m rerun_rlds_ur5.rlds --data path/to/rerun_rlds_ur5/tests/episode_yellow_rubber_duck.npy --type deligrasp`
