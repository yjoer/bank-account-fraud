import logging
import os
import subprocess
import sys

from crunchy_mining.util import set_low_priority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

experiments_clf = [
    # "sampling_v1",
    # "sampling_v2",
    # "sampling_v3",
    # "sampling_v4",
    # "sampling_v5",
    # "sampling_v6",
    # "preprocessing_v1",
    # "preprocessing_v2",
    # "preprocessing_v3",
    # "preprocessing_v4",
    # "preprocessing_v5",
    # "preprocessing_v6",
    # "preprocessing_v7",
    # "resampling_v1",
    # "resampling_v2",
    # "resampling_v3",
    # "resampling_v4",
    # "resampling_v5",
    # "resampling_v6",
    # "resampling_v7",
    # "resampling_v8",
    # "feature_selection_v1",
    # "feature_selection_v2",
    # "tuning_v1",
    # "tuning_v2",
    # "tuning_v3",
    # "tuning_v4",
    # "tuning_v5",
    # "tuning_v6",
    # "tuning_v7",
    # "tuning_v8",
    # "tuning_v9",
    # "tuning_v10",
    # "retrain_v1",
    # "retrain_v2",
    # "retrain_v3",
    # "retrain_v4",
    # "retrain_v5",
    # "retrain_v6",
    # "retrain_v7",
    # "retrain_v8",
    # "retrain_v9",
    # "retrain_v10",
]


for experiment in experiments_clf:
    env = os.environ.copy()
    env["CM_EXPERIMENT"] = experiment

    logger.info(f"Running experiment: {experiment}")

    process = subprocess.Popen([sys.executable, "notebook_guard.py"], env=env)
    set_low_priority(process.pid)
    process.wait()
