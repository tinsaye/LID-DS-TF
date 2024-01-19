import os
import pickle

from algorithms.building_block import BuildingBlock
from decision_engines.transformer import AnomalyScore
from evaluation.fluctuation_analysis.cluster.fluctuation_analysis_tf import parser
from evaluation.fluctuation_analysis.metrics import Metrics
from features.int_embedding import IntEmbeddingConcat
from algorithms.features.impl.max_score_threshold import MaxScoreThreshold
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.stream_sum import StreamSum
from algorithms.features.impl.syscall_name import SyscallName
from algorithms.ids import IDS
from dataloader.base_data_loader import BaseDataLoader
from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction
from dataloader.syscall import Syscall

from utils.checkpoint import ModelCheckPoint


class CachedTF(BuildingBlock):
    """
    Use the cached anomaly scores from the Transformer model and simulate TF and Thresholding
    Used to determine the CFA without running the whole training and validation process
    """

    def __init__(self,
                 input_vector: BuildingBlock,
                 anomaly_scores: dict,
                 threshold: float,
                 decider: bool = True, ):
        super().__init__()
        self._input_vector = input_vector
        self._anomaly_scores = anomaly_scores
        self.threshold = threshold
        self._decider = decider
        self._BuildingBlock__config["anomaly_scores"] = ""

    def _calculate(self, syscall: Syscall) -> bool:
        input_vector = self._input_vector.get_result(syscall)
        if input_vector is None:
            return False
        ano_score = self._anomaly_scores[input_vector]
        if self._decider:
            return ano_score > self.threshold
        else:
            return ano_score

    def depends_on(self) -> list:
        return [self._input_vector]

    def is_decider(self):
        return self._decider


learning_rate = 0.001
direction = Direction.OPEN
batch_size = 256


def main():
    _parser = parser()
    _parser.add_argument(
        '-per', dest='percent', action='store', type=int, required=True,
        help='Thresholding offset percent'
    )
    _parser.add_argument(
        '-ep', dest='epoch', action='store', type=int, required=True,
        help='Epoch'
    )
    _parser.add_argument(
        '-cpus', dest='cpus', action='store', type=int, required=True,
        help='CPUs'
    )
    _parser.add_argument(
        '-stream', dest='stream', action='store', type=int, required=True,
    )
    args = _parser.parse_args()

    print(args)
    dataset_base = args.base_path
    dataset = args.dataset
    scenario = args.scenario
    checkpoint_dir = args.checkpoint_dir
    ngram_length = args.ngram_length
    model_dim = args.model_dim
    layers = args.layers
    num_heads = args.num_heads
    custom_split = args.custom_split
    dropout = args.dropout
    run = args.run
    percent = args.percent
    epoch = args.epoch
    cpus = args.cpus
    stream = args.stream

    scenario_path = os.path.join(dataset_base, dataset, scenario)
    dataloader: BaseDataLoader = dataloader_factory(scenario_path, direction)

    if run > 0:
        checkpoint_dir = os.path.join(checkpoint_dir, f"run_{run}")

    checkpoint = ModelCheckPoint(
        scenario_name=scenario,
        models_dir=os.path.join(checkpoint_dir, "models"),
        lid_ds_version_name=dataset,
        algorithm="tf",
        algo_config={
            "ngram_length": ngram_length,
            "dropout": dropout,
            "model_dim": model_dim,
            "lr": learning_rate,
            "direction": direction,
            "split": custom_split,
            "batch_size": batch_size,
            "heads": num_heads,
            "layers": layers,
        },
    )

    if "IDS_ON_CLUSTER" in os.environ:
        path = os.path.join(
            "/home/sc.uni-leipzig.de/ta651pyga/lidds_wt/thesis/LID-DS/algorithms/evaluation/fluctuation_analysis/cluster/",
            "metrics",
            f"run_{run}" if run > 0 else "",
        )
    else:
        path = os.path.join(checkpoint_dir, "metrics")
    config = ("TF", dataset, scenario, ngram_length, model_dim, layers, num_heads, dropout, custom_split)
    stats_path = os.path.join(path, "" if stream == 0 else f"stream{stream}", f"{'_'.join((str(c) for c in config))}_{epoch}_{percent}_stats.pickle")
    if os.path.exists(stats_path):
        print(f"Skipping {stats_path}")
        print(f"{config} {epoch} {percent}")
        return

    metrics_path = os.path.join(path, f"{'_'.join((str(c) for c in config))}.pickle")
    with open(metrics_path, "rb") as f:
        metrics: dict[Metrics] = pickle.load(f)
        threshold = metrics[epoch].thr[percent]

    anomaly_scores = checkpoint.load_epoch(epoch)["anomaly_scores"][AnomalyScore.LOSS]

    # FEATURES
    sys_name = SyscallName()
    int_emb = IntEmbeddingConcat([sys_name])
    ngram = Ngram(
        feature_list=[int_emb],
        ngram_length=ngram_length,
        thread_aware=True,
    )
    decider = CachedTF(
        input_vector=ngram,
        anomaly_scores=anomaly_scores,
        threshold=threshold,
        decider=stream == 0,
    )
    if stream > 0:
        decider = StreamSum(decider, False, stream, False)
        decider = MaxScoreThreshold(decider)

    ids = IDS(
        data_loader=dataloader,
        resulting_building_block=decider,
        plot_switch=False,
    )

    if cpus > 1:
        performance = ids.detect_parallel()
    else:
        performance = ids.detect()
    stats = performance.get_results()

    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    with open(stats_path + '.json', "w") as f:
        import json
        json.dump(stats, f)


if __name__ == '__main__':
    main()
