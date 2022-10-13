import argparse
import os
import sys
import time
from pprint import pprint

from pandas import datetime

from algorithms.decision_engines.transformer import Transformer, AnomalyScore
from algorithms.features.impl.int_embedding import IntEmbedding
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.ids import IDS
from algorithms.persistance import save_to_json, print_as_table, ModelCheckPoint
from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction

lid_ds_versions = [
    "LID-DS-2019",
    "LID-DS-2021"
]
scenario_names = [
    "CVE-2017-7529",
    "CVE-2014-0160",
    "CVE-2012-2122",
    "Bruteforce_CWE-307",
    "CVE-2020-23839",

    "CWE-89-SQL-injection",
    "PHP_CWE-434",
    "ZipSlip",
    "CVE-2018-3760",
    "CVE-2020-9484",

    "EPS_CWE-434",
    "CVE-2019-5418",
    "Juice-Shop",
    "CVE-2020-13942",
    "CVE-2017-12635_6"
]


def _parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the Transformer based IDS ')

    parser.add_argument(
        '-d', dest='base_path', action='store', type=str, required=True,
        help='LID-DS base path'
    )
    parser.add_argument(
        '-v', dest='lid_ds_version', action='store', type=str, required=True,
        help='LID-DS version'
    )
    parser.add_argument(
        '-s', dest='scenario', action='store', type=str, required=False,
        help='Scenario name'
    )
    parser.add_argument(
        '-c', dest='checkpoint_dir', action='store', type=str, required=True,
        help='Models checkpoint base directory'
    )

    parser.add_argument(
        '-n', dest='ngram_length', action='store', type=int, required=True,
        help='Ngram length'
    )
    parser.add_argument(
        '-t', dest='thread_aware', action='store', type=bool, required=True,
        help='Thread aware ngrams'
    )

    parser.add_argument(
        '-f', dest='feedforward_dim', action='store', type=int, required=True,
        help='Thread aware ngrams'
    )
    parser.add_argument(
        '-l', dest='layers', action='store', type=int, required=True,
        help='Number of encoder and decoder layers'
    )
    parser.add_argument(
        '-m', dest='model_dim', action='store', type=int, required=True,
        help='TF model dimension (aka. emb size)'
    )

    return parser.parse_args()


def main():
    lid_ds_version_number = 1
    scenario_number = 0
    checkpoint_dir = "Models"
    retrain = False
    ngram_length = 11
    thread_aware = True

    anomaly_score = AnomalyScore.MEAN
    layers = 6
    model_dim = 8

    feedforward_dim = 1024
    batch_size = 256 * 2
    num_heads = 2
    epochs = 20
    dropout = 0.1

    if "IDS_ON_CLUSTER" in os.environ:
        args = _parse_args()
        scenario = args.scenario
        lid_ds_version = args.lid_ds_version
        scenario_path = args.base_path + scenario
        checkpoint_dir = args.checkpoint_dir

        ngram_length = args.ngram_length
        thread_aware = args.thread_aware
        layers = args.layers
        model_dim = args.model_dim
        feedforward_dim = args.feedforward_dim
        num_heads = model_dim / 4
    else:
        # getting the LID-DS base path from argument or environment variable
        if len(sys.argv) > 1:
            lid_ds_base_path = sys.argv[1]
        else:
            try:
                lid_ds_base_path = os.environ['LID_DS_BASE']
            except KeyError:
                raise ValueError(
                    "No LID-DS Base Path given. Please specify as argument or set Environment Variable "
                    "$LID_DS_BASE"
                )
        scenario = scenario_names[scenario_number]
        lid_ds_version = lid_ds_versions[lid_ds_version_number]
        scenario_path = f"{lid_ds_base_path}/{lid_ds_version}/{scenario}"

    # data loader for scenario
    dataloader = dataloader_factory(scenario_path, direction=Direction.OPEN)

    checkpoint = ModelCheckPoint(
        scenario_names[scenario_number],
        lid_ds_versions[lid_ds_version_number],
        "transformer",
        algo_config={
            "ngram_length": ngram_length,
            "thread_aware": thread_aware,
            "batch_size": batch_size,
            "layers": layers,
            "model_dim": model_dim,
            "feedforward_dim": feedforward_dim,
            "direction": dataloader.get_direction_string()
        },
        models_dir=checkpoint_dir
    )

    # embedding
    name = SyscallName()

    int_embedding = IntEmbedding()

    ngram = Ngram(
        feature_list=[int_embedding],
        thread_aware=thread_aware,
        ngram_length=ngram_length
    )

    distinct_syscalls = dataloader.distinct_syscalls_training_data()

    for epochs in reversed(range(5, 21, 5)):
        start = time.time()
        # decision engine (DE)
        transformer = Transformer(
            input_vector=ngram,
            distinct_syscalls=distinct_syscalls,
            retrain=retrain,
            epochs=epochs,
            batch_size=batch_size,
            anomaly_scoring=anomaly_score,
            checkpoint=checkpoint,
            layers=layers,
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            feedforward_dim=feedforward_dim,
        )

        # define the used features and train
        ids = IDS(
            data_loader=dataloader,
            resulting_building_block=transformer,
            plot_switch=False
        )

        # threshold
        ids.determine_threshold()

        performance = ids.detect()
        end = time.time()

        stats = performance.get_results()

        if stats is None:
            stats = {}
        stats['dataset'] = lid_ds_version
        stats['scenario'] = scenario
        stats['anomaly_score'] = anomaly_score.name
        stats['ngram'] = ngram_length
        stats['batch_size'] = batch_size
        stats['epochs'] = epochs
        stats['model_dim'] = model_dim
        stats['num_heads'] = num_heads
        stats['layers'] = layers
        stats['dropout'] = dropout
        stats['feedforward_dim'] = feedforward_dim
        stats['thread_aware'] = thread_aware
        stats['train_losses'] = transformer.train_losses
        stats['val_losses'] = transformer.val_losses
        stats['config'] = ids.get_config()
        stats['direction'] = dataloader.get_direction_string()
        stats['date'] = str(datetime.datetime.now().date())
        stats['duration'] = str(end - start)

        pprint(stats)
        result_path = f'{checkpoint.model_path_base}/{checkpoint.model_name}.json'
        save_to_json(stats, result_path)
        print_as_table(path=result_path)


if __name__ == '__main__':
    main()
