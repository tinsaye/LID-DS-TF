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
import pickle
from typing import Iterable

from algorithms.building_block import BuildingBlock
from algorithms.data_preprocessor import DataPreprocessor
from algorithms.features.impl.ngram import Ngram
from algorithms.features.impl.one_hot_encoding import OneHotEncoding
from algorithms.features.impl.syscall_name import SyscallName
from dataloader.base_data_loader import BaseDataLoader
from dataloader.dataloader_factory import dataloader_factory
from dataloader.direction import Direction
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from decision_engines.ae import AE
from decision_engines.transformer import Transformer, AnomalyScore
from evaluation.fluctuation_analysis.anomaly_scores import AnomalyScores
from evaluation.fluctuation_analysis.ngs import Ngs
from src.evaluation.fluctuation_analysis.ngrams_collector import NgramsCollector
from src.features.int_embedding import IntEmbeddingConcat
from utils.checkpoint import ModelCheckPoint


def collect_ngrams(ngram_bb: Ngram, scenario_path, direction: Direction, **kwargs) -> NgramsCollector:
    """
        Simulates the evaluation pipeline and collects the n-grams with a fake BuildingBlock.
        The collected n-grams can be cached for later use.
        This allows us to save a considerable amount of time when running the analysis
    Args:
        ngram_bb: the ngram building block
        scenario_path: path to the specific scenario of a dataset
        direction: system calls direction
        **kwargs: keywords arguments passed to the dataloader factory, necessary for ADFA dataset

    Returns:
        An instance of the NgramCollector
    """
    collector = NgramsCollector(ngram_bb)

    dataloader: BaseDataLoader = dataloader_factory(scenario_path, direction, **kwargs)
    data_preprocessor = DataPreprocessor(dataloader, collector)

    for recording in tqdm(dataloader.test_data()):
        if recording.metadata()["exploit"]:
            collector.recording_exploit(recording.name)
            current_exploit_time = recording.metadata()["time"]["exploit"][0]["absolute"]
            for syscall in recording.syscalls():
                collector.exploit_on(syscall)
                syscall_time = syscall.timestamp_unix_in_ns() * 1e-9
                if syscall_time < current_exploit_time:
                    collector.before_exploit_on(syscall)
                else:
                    collector.after_exploit_on(syscall)
        else:
            collector.recording_norm(recording.name)
            for syscall in recording.syscalls():
                collector.normal_on(syscall)
        data_preprocessor.new_recording()
    return collector


def ngram_sets(collector: NgramsCollector) -> Ngs:
    """
        Converts results from the NgramCollector to ngram sets
    Args:
        collector: the ngram collector

    Returns:
        An instance of Ngs containing the ngrams sets
    """
    train_set = list(collector.train_set_counts.keys())
    val_set = list(collector.val_set_counts.keys())
    exploit_set = list(collector.exploit_set_counts.keys())
    normal_set = list(collector.normal_set_counts.keys())
    before_exploit_set = list(collector.before_exploit_set_counts.keys())
    after_exploit_set = list(collector.after_exploit_set_counts.keys())
    val_exc_train = list(set(val_set) - set(train_set))
    exploit_exc_train = list(set(exploit_set) - set(train_set))
    normal_exc_train = list(set(normal_set) - set(train_set))
    exploit_exc_val = list(set(exploit_set) - set(val_set))
    normal_exc_val = list(set(normal_set) - set(val_set))
    exploit_exc_train_val = list(set(exploit_set) - (set(train_set) | set(val_set)))
    normal_exc_train_val = list(set(normal_set) - (set(train_set) | set(val_set)))
    before_exploit_exc_train_val = list(set(before_exploit_set) - (set(train_set) | set(val_set)))
    after_exploit_exc_train_val = list(set(after_exploit_set) - (set(train_set) | set(val_set)))
    before_exploit_exc_train = list(set(before_exploit_set) - set(train_set))
    after_exploit_exc_train = list(set(after_exploit_set) - set(train_set))
    train_set_split, val_set_split = train_test_split(
        list(train_set), test_size=0.2 - len(val_exc_train) / len(train_set), random_state=42
    )
    val_set_split = list(set(val_set_split) | set(val_exc_train))
    all_set = list(set(train_set) | set(val_set) | set(exploit_set) | set(normal_set))
    true_all_len = collector.train_set_length + collector.val_set_length + collector.exploit_set_length + collector.normal_set_length

    result = Ngs()
    result.train_set = train_set
    result.val_set = val_set
    result.exploit_set = exploit_set
    result.normal_set = normal_set
    result.before_exploit_set = before_exploit_set
    result.after_exploit_set = after_exploit_set
    result.val_exc_train = val_exc_train
    result.exploit_exc_train = exploit_exc_train
    result.normal_exc_train = normal_exc_train
    result.exploit_exc_val = exploit_exc_val
    result.normal_exc_val = normal_exc_val
    result.exploit_exc_train_val = exploit_exc_train_val
    result.normal_exc_train_val = normal_exc_train_val
    result.before_exploit_exc_train_val = before_exploit_exc_train_val
    result.after_exploit_exc_train_val = after_exploit_exc_train_val
    result.before_exploit_exc_train = before_exploit_exc_train
    result.after_exploit_exc_train = after_exploit_exc_train
    result.train_set_split = train_set_split
    result.val_set_split = val_set_split
    result.all_set = all_set
    result.true_all_len = true_all_len
    result.train_set_length = collector.train_set_length
    result.val_set_length = collector.val_set_length
    result.exploit_set_length = collector.exploit_set_length
    result.normal_set_length = collector.normal_set_length
    result.before_exploit_set_length = collector.before_exploit_set_length
    result.after_exploit_set_length = collector.after_exploit_set_length
    result.per_rec_after = collector.per_rec_after
    result.per_rec_before = collector.per_rec_before
    result.per_rec_normal = collector.per_rec_normal

    return result


def anomaly_scores_for_epoch(model, epoch, NGS: Ngs) -> AnomalyScores:
    """
       Trains the model (if necessary) or retrieves cached results and calculates anomaly scores for a given epoch
    Args:
        model: model building block
        epoch: number of epochs to train
        NGS: ngrams sets

    Returns:
        Instance of AnomalyScores
    """
    model._epochs = epoch
    model.load_epoch(epoch)

    anomaly_scores_all = {}
    if model.use_cache:
        anomaly_scores_all = model.get_cached_scores()
    if not len(anomaly_scores_all):
        anomaly_scores_all = model.batched_results(list(NGS.all_set), batch_size=4096)
        model.save_epoch(epoch)

    anomaly_scores_train = [anomaly_scores_all[ng] for ng in NGS.train_set]
    anomaly_scores_val = [anomaly_scores_all[ng] for ng in NGS.val_set]
    anomaly_scores_before_exploit = [anomaly_scores_all[ng] for ng in NGS.before_exploit_set]
    anomaly_scores_after_exploit = [anomaly_scores_all[ng] for ng in NGS.after_exploit_set]
    anomaly_scores_normal = [anomaly_scores_all[ng] for ng in NGS.normal_set]
    anomaly_scores_val_exc_train = [anomaly_scores_all[ng] for ng in NGS.val_exc_train]
    anomaly_scores_before_exploit_exc_train = [anomaly_scores_all[ng] for ng in NGS.before_exploit_exc_train]
    anomaly_scores_normal_exc_train = [anomaly_scores_all[ng] for ng in NGS.normal_exc_train]
    anomaly_scores_after_exploit_exc_train = [anomaly_scores_all[ng] for ng in NGS.after_exploit_exc_train]

    anomaly_scores_after_exploit_per_recording = {}
    for name, rec_ng in NGS.per_rec_after.items():
        if len(rec_ng) == 0:
            anomaly_scores_after_exploit_per_recording[name] = [0]
        for ng in rec_ng:
            score = anomaly_scores_all[ng]
            if name in anomaly_scores_after_exploit_per_recording:
                anomaly_scores_after_exploit_per_recording[name].append(score)
            else:
                anomaly_scores_after_exploit_per_recording[name] = [score]

    anomaly_scores_before_exploit_per_recording = {}
    for name, rec_ng in NGS.per_rec_before.items():
        if len(rec_ng) == 0:
            anomaly_scores_before_exploit_per_recording[name] = [0]
        for ng in rec_ng:
            score = anomaly_scores_all[ng]
            if name in anomaly_scores_before_exploit_per_recording:
                anomaly_scores_before_exploit_per_recording[name].append(score)
            else:
                anomaly_scores_before_exploit_per_recording[name] = [score]

    anomaly_scores_normal_per_recording = {}
    for name, rec_ng in NGS.per_rec_normal.items():
        if len(rec_ng) == 0:
            anomaly_scores_normal_per_recording[name] = [0]
        for ng in rec_ng:
            score = anomaly_scores_all[ng]
            if name in anomaly_scores_normal_per_recording:
                anomaly_scores_normal_per_recording[name].append(score)
            else:
                anomaly_scores_normal_per_recording[name] = [score]

    anomaly_scores = AnomalyScores(
        epoch,
        anomaly_scores_train,
        anomaly_scores_val,
        anomaly_scores_before_exploit,
        anomaly_scores_after_exploit,
        anomaly_scores_normal,
        anomaly_scores_val_exc_train,
        anomaly_scores_before_exploit_exc_train,
        anomaly_scores_after_exploit_exc_train,
        anomaly_scores_normal_exc_train,
        anomaly_scores_after_exploit_per_recording,
        anomaly_scores_before_exploit_per_recording,
        anomaly_scores_normal_per_recording, )

    return anomaly_scores


def roc_metrics_for_threshold_with_normal(anos: AnomalyScores):
    """
        Determines true positive rate and false positive rate based on anomaly scores
        Uses the threshold determined by the model.
        Includes the anomaly scores from the normal sets ngrams
    Args:
        anos: anomaly scores

    Returns:
        the true positive rate and false positive rate

    """
    threshold = anos.threshold
    num_rec = len(anos.after_exploit_per_recording)
    num_norm = len(anos.normal_per_recording)
    is_anomaly_after_per_recording = [any([sc > threshold for sc in _scores])
                                      for _scores in anos.after_exploit_per_recording.values()]
    is_anomaly_before_per_recording = [any([sc > threshold for sc in _scores])
                                       for _scores in anos.before_exploit_per_recording.values()]
    is_anomaly_normal_per_recording = [any([sc > threshold for sc in _scores])
                                       for _scores in anos.normal_per_recording.values()]

    fp = sum(is_anomaly_before_per_recording) + sum(is_anomaly_normal_per_recording)
    tp = sum(is_anomaly_after_per_recording)

    tn = num_rec + num_norm - fp
    fn = num_rec - tp

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr, tp, fp


def fp_tp_for_threshold(anos: AnomalyScores) -> tuple[int, int]:
    """
        Determines the number of true positives and false positives based on the given anomaly scores
    Args:
        anos: anomaly scores

    Returns:
        Number of false positives and true positives
    """
    threshold = anos.threshold
    is_anomaly_before_per_recording = [any([sc > threshold for sc in _scores])
                                       for _scores in anos.before_exploit_per_recording.values()]
    is_anomaly_after_per_recording = [any([sc > threshold for sc in _scores])
                                      for _scores in anos.after_exploit_per_recording.values()]

    fp = sum(is_anomaly_before_per_recording)
    tp = sum(is_anomaly_after_per_recording)

    return fp, tp


def aoc_metrics_for_epoch_with_normal(anos: AnomalyScores):
    """
        Determines true positive rates and false positive rates based on anomaly scores
        Uses different thresholds
        Includes the anomaly scores from the normal sets ngrams
    Args:
        anos: anomaly scores

    Returns:
        the true positive rate and false positive rate

    """
    num_rec = len(anos.before_exploit_per_recording)
    num_norm = len(anos.normal_per_recording)
    y_true = [0] * num_rec + [1] * num_rec + [0] * num_norm
    y_score = [max(scores) for scores in anos.before_exploit_per_recording.values()]
    y_score += [max(scores) for scores in anos.after_exploit_per_recording.values()]
    y_score += [max(scores) for scores in anos.normal_per_recording.values()]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return thresholds, tpr, fpr, roc_auc


def get_anomaly_scores_for_epochs(_model, epochs, _NGS, _collector, config: Iterable[int], base_path=""):
    anos_per_epoch = {}

    cache_path = f"anomaly_scores/{'_'.join((str(c) for c in config))}.pickle"
    cache_path = os.path.join(base_path, cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if _model.use_cache:
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                anos_per_epoch = pickle.load(f)
            return anos_per_epoch

    for epoch in tqdm(epochs):
        anos: AnomalyScores = anomaly_scores_for_epoch(_model, epoch, _NGS)
        anos_per_epoch[epoch] = anos

    with open(cache_path, "wb") as f:
        pickle.dump(anos_per_epoch, f)
    return anos_per_epoch


def cache_anomaly_scores(config: Iterable[int], anos_per_epoch, base_path=""):
    cache_path = f"anomaly_scores/{'_'.join((str(c) for c in config))}.pickle"
    cache_path = os.path.join(base_path, cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(anos_per_epoch, f)


def get_cached_anomaly_scores(config: Iterable[int], base_path=""):
    cache_path = f"anomaly_scores/{'_'.join((str(c) for c in config))}.pickle"
    cache_path = os.path.join(base_path, cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            anos_per_epoch = pickle.load(f)
        return anos_per_epoch
    else:
        print(f"no cache for {config}")
        return {}


def cache_losses(_model, config: Iterable[int], base_path=""):
    cache_path = f"losses/{'_'.join((str(c) for c in config))}.pickle"
    cache_path = os.path.join(base_path, cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        return
    losses = _model.train_losses, _model.val_losses
    with open(cache_path, "wb") as f:
        pickle.dump(losses, f)


def get_cached_losses(config: Iterable[int], base_path=""):
    cache_path = f"losses/{'_'.join((str(c) for c in config))}.pickle"
    cache_path = os.path.join(base_path, cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            losses = pickle.load(f)
        return losses
    else:
        print(f"no cache for {config}")
        return None, None


def prepare_tf_ngs(dataset_base,
                   ngram_length: int,
                   direction: Direction,
                   dataset: str,
                   scenario: str,
                   base_path="") -> NgramsCollector:
    """
        Prepares the ngrams for the transformer model evaluation
        Caches the ngrams to disc
    Args:
        dataset_base: base directory for the dataset
        ngram_length: ngram length
        direction: system call direction
        dataset: dataset name
        scenario: dataset scenario
        base_path: base path where the results (cached ngrams) should be stored

    Returns:
        An NgramCollector instance

    """
    path = f"ngrams/tf_{dataset}_{scenario}_{ngram_length}_{direction}.pickle"
    path = os.path.join(base_path, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "rb") as f:
            collector = pickle.load(f)
        print(f"skip {scenario}")
        return collector

    scenario_path = os.path.join(dataset_base, dataset, scenario)

    sys_name = SyscallName()
    int_emb = IntEmbeddingConcat([sys_name])
    ngram = Ngram(feature_list=[int_emb], ngram_length=ngram_length, thread_aware=True, )

    collector = collect_ngrams(ngram, scenario_path, direction)
    syscall_dict = {v: k for k, v in int_emb._encoding_dict[0].items()}
    syscall_dict[0] = "<unk>"
    ng_syscall = int_emb._encoding_dict[0]
    ng_syscall = {v: k for k, v in ng_syscall.items()}
    collector.syscall_dict = syscall_dict, ng_syscall

    with open(path, "wb") as f:
        pickle.dump(collector, f)

    return collector


def prepare_ae_ngs(dataset_base,
                   ngram_length: int,
                   direction: Direction,
                   dataset: str,
                   scenario: str,
                   base_path="") -> NgramsCollector:
    """
        Prepares the ngrams for the autoencoder model evaluation
        Caches the ngrams to disc
    Args:
        dataset_base: base directory for the dataset
        ngram_length: ngram length
        direction: system call direction
        dataset: dataset name
        scenario: dataset scenario
        base_path: base path where the results (cached ngrams) should be stored

    Returns:
        An NgramCollector instance

    """
    path = f"ngrams/ae_{dataset}_{scenario}_{ngram_length}_{direction}.pickle"
    path = os.path.join(base_path, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, "rb") as f:
            collector = pickle.load(f)
        print(f"skip {scenario}")
        return collector

    if os.path.exists(path):
        with open(path, "rb") as f:
            collector = pickle.load(f)
        print(f"skip {scenario}")
        return collector

    scenario_path = os.path.join(dataset_base, dataset, scenario)

    sys_name = SyscallName()
    ohe = OneHotEncoding(sys_name)
    ngram = Ngram(feature_list=[ohe], ngram_length=ngram_length, thread_aware=True, )

    collector = collect_ngrams(ngram, scenario_path, direction)
    syscall_dict = {v: k for k, v in ohe._input_to_int_dict.items()}
    syscall_dict[len(syscall_dict)] = "<unk>"
    ng_ohe = ohe._int_to_ohe_dict
    ng_ohe = {v: k for k, v in ng_ohe.items()}
    collector.syscall_dict = syscall_dict, ng_ohe

    with open(path, "wb") as f:
        pickle.dump(collector, f)

    return collector


def train_ae_model(scenario,
                   dataset,
                   ngram_length,
                   dropout,
                   learning_rate,
                   direction,
                   custom_split,
                   NGS: Ngs,
                   epochs=3000,
                   base_path=""):
    """ Train the autoencoder"""
    checkpoint = ModelCheckPoint(
        scenario_name=scenario,
        models_dir=os.path.join(base_path, "models"),
        lid_ds_version_name=dataset,
        algorithm="ae",
        algo_config=dict(
            ngram_length=ngram_length,
            dropout=dropout,
            learning_rate=learning_rate,
            direction=direction,
            split=custom_split
        ), )

    model = AE(
        input_vector=BuildingBlock(),  # Not needed, use fake
        epochs=epochs,
        dropout=dropout,
        use_early_stopping=False,
        checkpoint=checkpoint,
        learning_rate=learning_rate, )
    if not custom_split:
        model._training_set = NGS.train_set
        model._validation_set = NGS.val_set
    else:
        model._training_set = NGS.train_set_split
        model._validation_set = NGS.val_set_split

    model._input_size = len(NGS.train_set[0])
    model.fit()
    return model


def train_tf_model(scenario,
                   dataset,
                   ngram_length,
                   dropout,
                   learning_rate,
                   direction,
                   custom_split,
                   model_dim,
                   batch_size,
                   emb,
                   NGS: Ngs,
                   epochs,
                   num_heads=2,
                   layers=2,
                   base_path=""):
    """ Train the transformer model """
    checkpoint = ModelCheckPoint(
        scenario_name=scenario,
        models_dir=os.path.join(base_path, "models"),
        lid_ds_version_name=dataset,
        algorithm="tf",
        algo_config=dict(
            ngram_length=ngram_length,
            dropout=dropout,
            model_dim=model_dim,
            lr=learning_rate,
            direction=direction,
            split=custom_split,
            batch_size=batch_size,
            heads=num_heads,
            layers=layers
        ), )

    model = Transformer(
        input_vector=BuildingBlock(),  # Not needed, use fake
        epochs=epochs,
        anomaly_scoring=AnomalyScore.LOSS,
        batch_size=batch_size,
        num_heads=num_heads,
        layers=layers,
        model_dim=model_dim,
        dropout=dropout,
        feedforward_dim=model_dim * 4,
        pre_layer_norm=True,
        dedup_train_set=True,
        retrain=False,
        checkpoint=checkpoint,
        int_embedding=emb,
        learning_rate=learning_rate
    )

    if not custom_split:
        model._training_set = NGS.train_set
        model._validation_set = NGS.val_set
    else:
        model._training_set = NGS.train_set_split
        model._validation_set = NGS.val_set_split
    model.fit()
    return model