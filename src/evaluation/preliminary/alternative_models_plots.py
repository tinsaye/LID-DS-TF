"""
Plots from the preliminary experiments
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

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import proplot as pplt
from algorithms.evaluation.experiment_result_queries import ResultQuery
from matplotlib.ticker import FormatStrFormatter
from tabulate import tabulate

from evaluation.preliminary.common import config_aliases_sin_epochs, config_aliases_no_stream, features_no_stream, features, \
    config_aliases, _get_metrics, TRAIN, VAL, dataset_short, scenario_short, DR, THRESHOLD, FP

font = {
    'weight': 'bold',
    'size': 12
}

matplotlib.rc('font', **font)
plt.style.use('bmh')


def initial_repeated_old_repro():
    results = ResultQuery(collection_name="epoch_experiments_ae_rely_ohe").find_best_algorithm(
        algorithms=["AE"],
        group_by=["dataset", "scenario", "ng", "run"],
        features=features,
        sort_by={"scenario": 1, "dataset": 1, "algorithm": 1, "ng": 1, "run": 1},
        config_aliases=config_aliases_sin_epochs,
    )

    print(tabulate(results, headers="keys", tablefmt="github"))


def initial_repeated_old_repro_mlp():
    results = ResultQuery(collection_name="final_mlp_no_seed_repeat_early").find_best_algorithm(
        algorithms=["MLP"],
        group_by=["dataset", "scenario", "ng", "run"],
        features=features,
        sort_by={"scenario": 1, "dataset": 1, "algorithm": 1, "ng": 1, "run": 1},
        config_aliases=config_aliases_sin_epochs,
    )

    print(tabulate(results, headers="keys", tablefmt="github"))


def initial_repeated_old_repro_mlp_no_stream():
    results = ResultQuery(collection_name="final_mlp_no_seed_repeat_early_no_stream").find_best_algorithm(
        algorithms=["MLP"],
        group_by=["dataset", "scenario", "ng", "run"],
        features=features_no_stream,
        sort_by={"scenario": 1, "dataset": 1, "algorithm": 1, "ng": 1, "run": 1},
        config_aliases=config_aliases_no_stream,
    )

    print(tabulate(results, headers="keys", tablefmt="github"))


def ae_mlp_epoch_analysis_picked():
    configs_dict = {
        "AE": [
            ("CVE-2017-7529", 5, 0.5, 0),
            ("CVE-2012-2122", 10, 0.5, 0),
            ("CVE-2018-3760", 5, 0.5, 0),
        ],
        "MLP": [
            ("CVE-2017-7529", 7, 0.5, 0),
            ("CVE-2019-5418", 7, 0.5, 1),
            ("Bruteforce_CWE-307", 7, 0.5, 1),
        ]}

    dataset = "LID-DS-2019"
    NUM_ROWS = 4
    fig, axs = pplt.subplots(
        top=0.5,
        right=2.2,
        left=3.5,
        bottom=0.5,
        hspace=0.5,
        nrows=NUM_ROWS,
        ncols=3,
        refwidth=3,
        refheight=2,
        # sharey=True,
        sharey=False,
        sharex=True,
        innerpad=0.2
    )
    idx0 = 0
    for algo, configs in configs_dict.items():
        max_fp = 1
        if algo == "AE":
            query = ResultQuery(collection_name="epoch_experiments_ae_no_seed_drop_05")
        else:
            query = ResultQuery(collection_name="final_mlp_no_seed_epochs_stream")
        for idx, (scenario, ngram, drop, run) in enumerate(configs):
            where = {
                "dataset": dataset,
                "scenario": scenario,
                "drop": drop,
                "$or": [{"ver": run}, {"run": run}],
                "ng": ngram if algo == "AE" else ngram + 1,
            }

            results = query.find_results(
                algorithms=[algo],
                scenarios=[scenario],
                datasets=[dataset],
                features=features,
                config_aliases=config_aliases,
                where=where,
            )

            if len(results) == 0:
                print("No results found for", where)
                continue

            detection_rates, false_positives, ths, epochs_x, train_losses, val_losses = _get_metrics(
                results,
                scale_ths=True
            )
            idx1 = (0 + idx0, idx)
            idx2 = (1 + idx0, idx)
            losses_x = range(1, len(train_losses) + 1)

            ln = axs[idx1].plot(
                losses_x,
                train_losses,
                **TRAIN
            )
            ln += axs[idx1].plot(
                losses_x,
                val_losses,
                **VAL
            )

            axs[idx1].set_xlim([-1, max(losses_x)])
            axs[idx1].yaxis.grid(which="both")
            axs[idx1].yaxis.set_major_formatter(FormatStrFormatter('%.0f' if algo == "AE" else '%.2f'))
            axs[idx1].format(
                title=f"{algo} {dataset_short[dataset]}-{scenario_short[scenario]} n{ngram} d{drop}",
                titleloc='ur'
            )

            axs[idx1].set_yticks([0, 5] if algo == "AE" else [0.015, 0.05])
            axs[idx1].set_ylim([0, 5] if algo == "AE" else [0.015, 0.052])

            if idx != 0:
                axs[idx1].set_yticklabels([])

            ln1 = axs[idx2].plot(
                epochs_x,
                detection_rates,
                **DR,
            )
            ln11 = axs[idx2].plot(
                epochs_x,
                ths,
                **THRESHOLD,
            )

            axs[idx2].set_xlim([-50, max(epochs_x)])
            axs[idx2].set_ylim([0, max(ths) * 1.05])
            axs[idx2].set_yticks([0, max(ths)])
            axs[idx2].yaxis.grid(which="both")
            if idx != 0:
                axs[idx2].set_yticklabels([])
            ax12 = axs[idx2].twinx()
            ln12 = ax12.plot(epochs_x, false_positives, **FP)

            max_fp = max(max(false_positives), max_fp)
            ax12.set_ylim([0, max(max(false_positives), 1) * 1.05])
            if idx == 2:
                ax12.set_ylim([0, max_fp * 1.05])
                ax12.set_yticks([0, max_fp])
                ax12.set_ylabel("CFA")
            else:
                ax12.set_yticks([])
            ax12.format(
                ylabelpad=-6,
                ylabelsize=12,
            )

        idx0 += 2
    title = f"{dataset}"

    # legend
    axs.format(
        ylabelpad=2,
        ylabelsize=12,
    )
    axs[-1, :].format(xlabel="Epochs")
    axs[0, 0].format(ylabel="Loss")
    axs[2, 0].format(ylabel="Loss", ylabelpad=-10)
    axs[1, 0].format(ylabel="DR / THR")
    axs[3, 0].format(ylabel="DR / THR")
    fig.legend(
        [ln[0], ln[1], ln1[0], ln11[0], ln12[0]],
        labels=["Training Loss", "Validation Loss", "Detection Rate", "Threshold", "CFA"],
        loc='b',
        pad=1,
        ncol=5,
        fontsize=20,
        frame=False,
    )

    file_path = f"alternative/epochs_dedup/"
    file_suffix = "_OTHERS"
    file_prefix = "picked_dedup_"
    path = f"{file_path}{file_prefix}{title.replace(' ', '_')}{file_suffix}.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=400, bbox_inches='tight')
    pplt.show()


if __name__ == '__main__':
    print("########## AE ##########")
    initial_repeated_old_repro()

    print("########## MLP ##########")
    initial_repeated_old_repro_mlp()

    print("########## MLP NO STREAM ##########")
    initial_repeated_old_repro_mlp_no_stream()
