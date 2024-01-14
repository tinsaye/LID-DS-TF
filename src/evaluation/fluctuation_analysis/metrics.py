from numpy import mean

from evaluation.fluctuation_analysis.anomaly_scores import AnomalyScores


class Metrics:
    """
        Collects all possible metrics from a full experiment run at different thresholds
    """
    def __init__(self, ano_scores: AnomalyScores):
        self.threshold = ano_scores.threshold
        self.threshold_train = ano_scores.threshold_train
        self.epoch = ano_scores.epoch
        self.val_loss = mean(ano_scores.val).item()
        self.train_loss = mean(ano_scores.train).item()
        self.thr = {}
        self.tpr = {}
        self.fpr = {}
        self.tp = {}
        self.fp = {}
        self.tpr = {}
        self.fpr = {}
        self.tp_ngs = {}
        self.precision = {}
        self.precision_ngs = {}
        self.fp_ngs = {}
        self.f1 = {}
        self.f1_ngs = {}
        self.stats = {}

        for per in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            max_val = self.threshold
            range_val = max_val - min(ano_scores.val)
            thr = max_val + range_val * per / 100
            self.thr[per] = thr
            self.tpr[per], self.fpr[per], self.tp[per], self.fp[per], self.tp_ngs[per], self.fp_ngs[per], \
                self.precision[per], self.precision_ngs[per], self.f1[per], self.f1_ngs[per] = metrics_for_threshold(
                ano_scores, thr
            )


def metrics_for_threshold(anos: AnomalyScores, threshold):
    """
    Calculate all metrics from the ng-sets of anomaly scores
    Args:
        anos: anomaly scores container
        threshold: threshold to be used

    Returns:
        tpr, fpr, tp, fp, tp_ngs, fp_ngs, precision, precision_ngs, f1, f1_ngs: metrics

    """
    num_rec = len(anos.after_exploit_per_recording)
    num_norm = len(anos.normal_per_recording)
    is_anomaly_after_per_recording = [any([sc > threshold for sc in _scores])
                                      for _scores in anos.after_exploit_per_recording.values()]
    is_anomaly_before_per_recording = [any([sc > threshold for sc in _scores])
                                       for _scores in anos.before_exploit_per_recording.values()]
    is_anomaly_normal_per_recording = [any([sc > threshold for sc in _scores])
                                       for _scores in anos.normal_per_recording.values()]
    tp_ngs = sum([score > threshold for score in anos.after_exploit])
    fp_ngs = sum(
        [score > threshold for score in anos.normal] + [score > threshold for score in anos.before_exploit]
    )
    fp = sum(is_anomaly_before_per_recording) + sum(is_anomaly_normal_per_recording)
    tp = sum(is_anomaly_after_per_recording)

    tn = num_rec + num_norm - fp
    fn = num_rec - tp

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if tp + fp_ngs == 0:
        precision_ngs = 0
    else:
        precision_ngs = tp / (tp + fp_ngs)

    recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    if precision_ngs + recall == 0:
        f1_ngs = 0
    else:
        f1_ngs = 2 * (precision_ngs * recall) / (precision_ngs + recall)
    return tpr, fpr, tp, fp, tp_ngs, fp_ngs, precision, precision_ngs, f1, f1_ngs
