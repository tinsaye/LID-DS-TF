""" Configurations and common functions """
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

import matplotlib.pyplot as plt
import proplot as pplt


pplt.rc.update({'fontsize': 11})
plt.style.use('bmh')
plt.rcParams['axes.facecolor'] = '#fbfbfb'

TRAIN_COL = 'blue'
VAL_COL = 'orange'
THR_COL = 'black'
DR_COL = 'green'
FP_COL = 'red'

TRAIN = {
    "color": TRAIN_COL,
    "linestyle": "-.",
    "linewidth": 1.5,
}
VAL = {
    "color": VAL_COL,
    "linestyle": "-.",
    "linewidth": 1.5,
}
THRESHOLD = {
    "color": THR_COL,
    "linestyle": "",
    "marker": "x",
    "markersize": 4,
}
DR = {
    "color": DR_COL,
    "linestyle": "--",
    "linewidth": 0.8,
    "marker": "*",
}
FP = {
    "color": FP_COL,
    "linestyle": "",
    "marker": "o",
    "markersize": 4,
}

config_aliases = {
    "Transformer": {
        "name": "MaxScoreThreshold",
        "input": [
            {
                "name": "Transformer",
                "epochs": "epochs",
                "num_heads": "heads",
                "layers": "layers",
                "model_dim": "dim",
                "language_model": "lm",
                "anomaly_scoring": "anomaly_scoring",
                "batch_size": "batch_size",
                "input": [
                    {
                        "name": "Ngram",
                        "@has": "NG",
                        "ngram_length": "ng",
                    }
                ],
            }
        ],
    },
    "AE": {
        "name": "MaxScoreThreshold",
        "input": [
            {
                "name": "AE",
                "epochs": "epochs",
                "dropout": "drop",
                "input": [
                    {
                        "name": "Ngram",
                        "ngram_length": "ng",
                    }
                ],
            }
        ],
    },
    "MLP": {
        "name": "MaxScoreThreshold",
        "input": [
            {
                "name": "StreamSum",
                "input": [{
                    "name": "MLP",
                    "epochs": "epochs",
                    "dropout": "drop",
                    "input": [{
                        "name": "NgramMinusOne",
                        "input": [{
                            "name": "Ngram",
                            "ngram_length": "ng",
                        }]
                    }]
                }],
            }
        ],
    },
}
features = {
    "Transformer": ["MaxScoreThreshold", "Transformer"],
    "AE": ["MaxScoreThreshold", "AE"],
    "MLP": ["MaxScoreThreshold", "StreamSum", "MLP"],
}
features_no_stream = {
    "MLP": ["MaxScoreThreshold", "MLP", "NgramMinusOne"],
}
datasets = [
    "LID-DS-2019",
    "LID-DS-2021"
]
dataset_short = {
    "LID-DS-2019": "19",
    "LID-DS-2021": "21",
}
scenario_short = {
    "CVE-2017-7529": "2017",
    "CVE-2014-0160": "2014",
    "CVE-2012-2122": "2012",
    "Bruteforce_CWE-307": "BF",
    "CWE-89-SQL-injection": "SQL",
    "SQL_Injection_CWE-89": "SQL",
    "PHP_CWE-434": "PHP",
    "CVE-2018-3760": "2018",
    "CVE-2019-5418": "2019",
}
config_aliases_sin_epochs = {
    "AE": {
        "name": "MaxScoreThreshold",
        "input": [
            {
                "name": "AE",
                "epochs": "epochs",
                "input": [
                    {
                        "name": "Ngram",
                        "ngram_length": "ng",
                    }
                ],
            }
        ],
    },
    "MLP": {
        "name": "MaxScoreThreshold",
        "input": [
            {
                "name": "StreamSum",
                "epochs": "epochs",
                "input": [
                    {
                        "name": "MLP",
                        "epochs": "epochs",
                        "input": [{
                            "name": "NgramMinusOne",
                            "input": [
                                {
                                    "name": "Ngram",
                                    "ngram_length": "ng",
                                }
                            ], }
                        ]
                    }
                ],

            }
        ],
    },
}
config_aliases_no_stream = {
    "MLP": {
        "name": "MaxScoreThreshold",
        "input": [
            {
                "name": "MLP",
                "epochs": "epochs",
                "input": [{
                    "name": "NgramMinusOne",
                    "input": [
                        {
                            "name": "Ngram",
                            "ngram_length": "ng",
                        }
                    ], }
                ]
            }
        ],
    },
}


def _get_metrics(results: list[dict], scale_ths=False):
    if not len(results):
        return
    results = sorted(results, key=lambda d: d['epochs'])
    detection_rates = [r["detection_rate"] for r in results]
    false_positives = [r["false_positives"] for r in results]
    if "threshold" in results[0]:
        ths = [r["threshold"] for r in results]
        if scale_ths:
            ths_max = max(ths)
            ths = [t / ths_max for t in ths]
    else:
        ths = []

    epochs_x = [r["epochs"] for r in results]
    if "train_losses" in results[-1]:
        train_losses = results[-1]["train_losses"].values()
        val_losses = results[-1]["val_losses"].values()
    else:
        train_losses = results[-1]["training_losses"].values()
        val_losses = results[-1]["validation_losses"].values()
    train_losses, val_losses = list(train_losses), list(val_losses)
    return detection_rates, false_positives, ths, epochs_x, train_losses, val_losses
