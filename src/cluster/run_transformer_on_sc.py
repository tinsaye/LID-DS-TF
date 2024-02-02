"""
script to start multiple jobs in cluster
"""
#  LID-DS-TF Copyright (c) 2024. Tinsaye Abye
#
#  LID-DS-TF is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  LID-DS-TF is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with LID-DS-TF.  If not, see <https://www.gnu.org/licenses/>.

import os
import time

scenario_2019 = [
    "Bruteforce_CWE-307",
    "CVE-2012-2122",
    "CVE-2014-0160",
    "CVE-2017-7529",
    "CVE-2018-3760",
    "CVE-2019-5418",
    "SQL_Injection_CWE-89",
    "EPS_CWE-434",
    "PHP_CWE-434",
    "ZipSlip",
]

scenario_2021 = [
    "Bruteforce_CWE-307",
    "CVE-2012-2122",
    "CVE-2014-0160",
    "CVE-2017-12635_6",
    "CVE-2017-7529",
    "CVE-2018-3760",
    "CVE-2019-5418",
    "CVE-2020-13942",
    "CVE-2020-23839",
    "CVE-2020-9484",
    "CWE-89-SQL-injection",
    "EPS_CWE-434",
    "Juice-Shop"
    "PHP_CWE-434",
    "ZipSlip",
]

EVAL = False
REPEAT = 1
USER = "ta651pyga"
DATASET = "LID-DS-2019"
CHECKPOINT_DIR = f"/work/users/{USER}/models"
NGRAM_LENGTHS = [4, 8, 16, 32]
thread_aware = True
language_model = True
LAYERS = [2, 4]
MODEL_DIMS = [8, 16, 32]
# NUM_HEADS = [1, 2, 4, 6]
batch_size = 256
DEDUP = True
ANOMALY_SCORINGS = ['LOSS']
USE_RET_VALUE = False
USE_PROCESS_NAME = False
USE_PATHS = False
USE_TIME_DELTA = False

BASE_PATH = f"/work/users/{USER}/datasets/{DATASET}/"
if '2019' in BASE_PATH:
    SCENARIOS = scenario_2019
else:
    SCENARIOS = scenario_2021
SCRIPT = 'run_tf_on_cluster_cpu.sh'

MAX_JOBS_IN_QUEUE = 600
NUM_EXPERIMENTS = 0


def count_queue():
    """
    counts the number of my jobs in the queue
    """
    return int(os.popen(f"squeue -u {USER} | wc -l").read().strip("\n")) - 1


def start_job(job_str):
    """
    starts the job given by str
    if the number of my jobs in the queue is smaller than MAX_JOBS_IN_QUEUE
    """
    while True:
        time.sleep(0.5)
        # get the number of jobs in the queue
        count = count_queue()
        if count < MAX_JOBS_IN_QUEUE:
            print(job_str)
            os.system(job_str)
            print(f"there are {count} jobs in queue")
            break


# start jobs for specified configurations
for run in range(REPEAT):
    for scenario in SCENARIOS:
        for ngram_length in NGRAM_LENGTHS:
            for layers in LAYERS:
                num_heads = layers
                for model_dim in MODEL_DIMS:
                    for anomaly_score in ANOMALY_SCORINGS:
                        ff_dim = model_dim * 4  # for now use recommended dimensions
                        if EVAL:
                            SCRIPT = 'run_tf_on_cluster_cpu.sh'
                            for epochs in reversed(range(60, 901, 60)):
                                NUM_EXPERIMENTS += 1
                                command = f"sbatch --job-name=ex_{NUM_EXPERIMENTS:05}{scenario}m{model_dim}l{layers}f{ff_dim}h{num_heads}lm{language_model}n{ngram_length}ret{USE_RET_VALUE}e{epochs} " + \
                                          f"{SCRIPT} " + \
                                          f"{BASE_PATH} " + \
                                          f"{DATASET} " + \
                                          f"{scenario} " + \
                                          f"{CHECKPOINT_DIR} " + \
                                          f"{ngram_length} " + \
                                          f"{thread_aware} " + \
                                          f"{ff_dim} " + \
                                          f"{layers} " + \
                                          f"{model_dim} " + \
                                          f"{num_heads} " + \
                                          f"{language_model} " + \
                                          f"{batch_size} " + \
                                          f"{DEDUP} " + \
                                          f"{anomaly_score} " + \
                                          f"{USE_RET_VALUE} " + \
                                          f"{epochs} " + \
                                          f"{USE_PROCESS_NAME} " + \
                                          f"{USE_PATHS} " + \
                                          f"{USE_TIME_DELTA} " + \
                                          f"{run} "

                                start_job(command)
                        else:
                            SCRIPT = 'run_tf_on_cluster.sh'
                            NUM_EXPERIMENTS += 1
                            command = f"sbatch --job-name=ex_{NUM_EXPERIMENTS:05}{scenario}m{model_dim}l{layers}f{ff_dim}h{num_heads}lm{language_model}n{ngram_length}ret{USE_RET_VALUE} " + \
                                      f"{SCRIPT} " + \
                                      f"{BASE_PATH} " + \
                                      f"{DATASET} " + \
                                      f"{scenario} " + \
                                      f"{CHECKPOINT_DIR} " + \
                                      f"{ngram_length} " + \
                                      f"{thread_aware} " + \
                                      f"{ff_dim} " + \
                                      f"{layers} " + \
                                      f"{model_dim} " + \
                                      f"{num_heads} " + \
                                      f"{language_model} " + \
                                      f"{batch_size} " + \
                                      f"{DEDUP} " + \
                                      f"{anomaly_score} " + \
                                      f"{USE_RET_VALUE} " + \
                                      f"{USE_PROCESS_NAME} " + \
                                      f"{USE_PATHS} " + \
                                      f"{USE_TIME_DELTA} " + \
                                      f"{run} "

                            start_job(command)

print(f"NUM_EXPERIMENTS = {NUM_EXPERIMENTS}")
