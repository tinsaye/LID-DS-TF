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

import pickle
from pathlib import Path

import proplot as pplt
from algorithms.evaluation.experiment_result_queries import ResultQuery
from matplotlib.ticker import FormatStrFormatter

from evaluation.preliminary.common import TRAIN, VAL, THRESHOLD, DR, FP, config_aliases, features, dataset_short, \
    scenario_short, \
    _get_metrics


def _picked(collection_name,
            configs: list[tuple],
            title,
            file_path,
            file_prefix,
            file_suffix="",
            NUM_ROWS=2,
            scale_ths=False):
    all_metrics = {}
    if Path(f"{file_path}/picked_metrics.pkl").exists():
        with open(f"{file_path}/picked_metrics.pkl", "rb") as f:
            print("Loading metrics from cache")
            all_metrics = pickle.load(f)

    query = ResultQuery(collection_name)
    fig, axs = pplt.subplots(
        top=0.5,
        right=2.2,
        left=3,
        bottom=0.5,
        hspace=0.5,
        nrows=NUM_ROWS,
        ncols=3,
        refwidth=2.7,
        refheight=2,
        sharey=False,
        sharex=True,
        innerpad=0,
    )
    max_fp = 1
    for idx, (dataset, scenario, heads, layers, dim, ngram) in enumerate(configs):
        where = {
            "dataset": dataset,
            "scenario": scenario,
            "layers": layers,
            "heads": heads,
            "dim": dim,
            "ng": ngram,
        }

        results = query.find_results(
            algorithms=["Transformer"],
            scenarios=[scenario],
            datasets=[dataset],
            features=features,
            config_aliases=config_aliases,
            where=where,
        )

        if len(results) == 0:
            print("No results found for", where)
            continue

        if (dataset, scenario, heads, layers, dim, ngram) in all_metrics:
            metrics = all_metrics[(dataset, scenario, heads, layers, dim, ngram)]
        else:
            metrics = _get_metrics(results, scale_ths)
            all_metrics[(dataset, scenario, heads, layers, dim, ngram)] = metrics

        detection_rates, false_positives, ths, epochs_x, train_losses, val_losses = metrics
        idx1 = (0, idx)
        idx2 = (1, idx)
        losses_x = range(1, len(train_losses) + 1)

        ln = axs[idx1].plot(
            losses_x,
            train_losses,
            **TRAIN,
        )
        ln += axs[idx1].plot(
            losses_x,
            val_losses,
            **VAL,
        )

        axs[idx1].set_xlim([0, max(losses_x)])
        min_loss = min(min(train_losses), min(val_losses))
        max_loss = max(max(train_losses), max(val_losses))
        axs[idx1].set_ylim([min_loss - min_loss * 0.05, max_loss])
        axs[idx1].set_yticks([min_loss, max_loss])
        axs[idx1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[idx1].format(
            title=f"{dataset_short[dataset]}-{scenario_short[scenario]} h{heads} l{layers} m{dim} n{ngram}",
            titleloc='ur'
        )
        if idx != 0:
            axs[idx1].set_yticklabels([])
        ln1 = axs[idx2].plot(epochs_x, detection_rates, **DR)
        ln11 = axs[idx2].plot(epochs_x, ths, **THRESHOLD)
        axs[idx2].set_xlim([0, max(epochs_x)])
        axs[idx2].set_ylim([0, max(ths) * 1.05])
        axs[idx2].set_yticks([0, max(ths)])
        if idx != 0:
            axs[idx2].set_yticklabels([])
        axs[idx2].yaxis.grid(which="both")
        ax12 = axs[idx2].twinx()
        ln12 = ax12.plot(
            epochs_x,
            false_positives,
            **FP,
        )
        max_fp = max(max(false_positives), max_fp)
        ax12.set_ylim([0, max(max(false_positives), 1) * 1.05])
        if idx == 2:
            ax12.set_ylim([0, max_fp * 1.05])
            ax12.set_yticks([0, max_fp])
            ax12.set_ylabel("CFA")
        else:
            ax12.set_yticks([])
        ax12.format(ylabelpad=-6, ylabelsize=12)

    # legend
    axs[-1, :].format(xlabel="Epochs")
    axs[0, 0].format(ylabel="Loss", ylabelpad=-4)
    axs[1, 0].format(ylabel="DR / THR")

    axs.format(
        ylabelsize=12,
        xlabelsize=12,
    )
    axs.format(
        abc='A.',
        abcloc='ul',
        abcweight='bold',
    )
    fig.legend(
        [ln[0], ln[1], ln1[0], ln11[0], ln12[0]],
        labels=["Training Loss", "Validation Loss", "Detection Rate", "Threshold", "CFA"],
        loc='b',
        pad=1,
        ncol=5,
        fontsize=20,
        frame=False,
    )

    file_path = f"{file_path}/"
    file_prefix = f"{file_prefix}"
    path = f"{file_path}{file_prefix}_{title.replace(' ', '_')}{file_suffix}.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    pplt.show()
    pplt.close(fig)
    fig.savefig(path, dpi=400, bbox_inches='tight')

    # cache metrics
    with open(f"{file_path}/picked_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)


# noinspection DuplicatedCode
def _picked_rest(collection_name,
                 configs: list[tuple],
                 title,
                 file_path,
                 file_prefix,
                 file_suffix="",
                 NUM_ROWS=2,
                 scale_ths=False):
    all_metrics = {}
    if Path(f"{file_path}/picked_metrics.pkl").exists():
        with open(f"{file_path}/picked_metrics.pkl", "rb") as f:
            print("Loading metrics from cache")
            all_metrics = pickle.load(f)

    query = ResultQuery(collection_name)
    fig, axs = pplt.subplots(
        top=0.5,
        right=2.5,
        left=2.5,
        bottom=0.5,
        hspace=0.5,
        nrows=NUM_ROWS,
        ncols=3,
        refwidth=2.7,
        refheight=2,
        sharey=True,
        sharex=True,
        innerpad=0.1,
    )
    max_fp = 1
    for idx, (dataset, scenario, heads, layers, dim, ngram) in enumerate(configs):
        where = {
            "dataset": dataset,
            "scenario": scenario,
            "layers": layers,
            "heads": heads,
            "dim": dim,
            "ng": ngram,
        }

        results = query.find_results(
            algorithms=["Transformer"],
            scenarios=[scenario],
            datasets=[dataset],
            features=features,
            config_aliases=config_aliases,
            where=where,
        )

        if len(results) == 0:
            print("No results found for", where)
            continue

        if (dataset, scenario, heads, layers, dim, ngram) in all_metrics:
            metrics = all_metrics[(dataset, scenario, heads, layers, dim, ngram)]
        else:
            metrics = _get_metrics(results, scale_ths)
            all_metrics[(dataset, scenario, heads, layers, dim, ngram)] = metrics

        detection_rates, false_positives, ths, epochs_x, train_losses, val_losses = metrics
        idx2 = idx
        axs[idx2].format(
            title=f"{dataset_short[dataset]}-{scenario_short[scenario]} h{heads} l{layers} m{dim} n{ngram}",
            titleloc='lc'
        )

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
        axs[idx2].set_xlim([0, max(epochs_x)])
        axs[idx2].set_ylim([0, max(ths) * 1.05])
        axs[idx2].set_yticks([0, max(ths)])
        axs[idx2].yaxis.grid(which="both")
        axs[idx2].set_ylabel("DR / THR")
        ax12 = axs[idx2].twinx()
        ln12 = ax12.plot(epochs_x, false_positives, **FP)
        max_fp = max(max(false_positives), max_fp)
        ax12.set_ylim([0, max(max(false_positives), 1) * 1.05])
        if (idx + 1) % 3 == 0:
            ax12.set_ylim([0, max_fp * 1.05])
            ax12.set_yticks([0, max_fp])
            ax12.set_ylabel("CFA")
            max_fp = 1
        else:
            ax12.set_yticks([])
        ax12.format(
            ylabelpad=-6,
            ylabelsize=12,
        )

    axs[-1, :].format(xlabel="Epochs")
    axs.format(ylabelsize=12, xlabelsize=12, )
    axs.format(
        abc='A.',
        abcloc='ul',
        abcweight='bold',
    )
    fig.legend(
        [ln1[0], ln11[0], ln12[0]],
        labels=["Detection Rate", "Threshold", "CFA"],
        loc='b',
        pad=1,
        ncol=5,
        fontsize=20,
        frame=False,
    )

    file_path = f"{file_path}/"
    file_prefix = f"{file_prefix}"
    path = f"{file_path}{file_prefix}_{title.replace(' ', '_')}{file_suffix}.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    pplt.show()
    fig.savefig(path, dpi=400, bbox_inches='tight')

    # cache metrics
    with open(f"{file_path}/picked_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)


def pre_dedup_picked_with_loss():
    configs_dict = [
        ## ("LID-DS-2019", "CVE-2017-7529", 4, 4, 8, 4),
        # ("LID-DS-2019","CVE-2017-7529", 2, 2, 8, 4),
        # ("LID-DS-2019","CVE-2017-7529", 2, 2, 8, 16),
        ("LID-DS-2019", "CVE-2017-7529", 2, 2, 32, 4),
        # ("LID-DS-2019","CVE-2017-7529", 2, 2, 32, 16),
        # ("LID-DS-2019","CVE-2017-7529", 4, 4, 8, 4),
        # ("LID-DS-2019","CVE-2017-7529", 4, 4, 8, 16),
        # ("LID-DS-2019","CVE-2017-7529", 4, 4, 32, 4),
        # ("LID-DS-2019","CVE-2017-7529", 4, 4, 32, 16),
        # ("LID-DS-2019","CVE-2017-7529", 4, 4, 8, 16),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 4, 4, 32, 4),
        ("LID-DS-2019", "CVE-2012-2122", 4, 4, 8, 16),
        # ("LID-DS-2019", "CVE-2012-2122", 2, 2, 8, 16),
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 32, 16),
        # ("LID-DS-2019","PHP_CWE-434", 2, 2, 8, 16),
        ("LID-DS-2019", "SQL_Injection_CWE-89", 2, 2, 8, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 2, 2, 32, 4),
        # ("LID-DS-2021","CVE-2017-7529", 2, 2, 8, 16),
        # ("LID-DS-2021","CVE-2014-0160", 4, 4, 32, 4),
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 32, 16),
        ## ("LID-DS-2021", "Bruteforce_CWE-307", 4, 4, 32, 16),
        # ("LID-DS-2021","PHP_CWE-434", 2, 2, 32, 16),
        # ("LID-DS-2021","CWE-89-SQL-injection", 2, 2, 8, 4),
        # ("LID-DS-2021","CWE-89-SQL-injection", 2, 2, 32, 4),
        # ("LID-DS-2021","CWE-89-SQL-injection", 4, 4, 32, 4),
    ]

    title = f"BOTH"
    file_suffix = ""
    file_path = "transformer/epochs_pre_dedup/"
    file_prefix = f"picked_pre_dedup_with_loss"
    _picked(
        "final_predup", configs_dict, title, file_path, file_prefix, file_suffix
    )


def pre_dedup_picked_rest():
    configs_dict = [
        # ("LID-DS-2019","CVE-2017-7529", 4, 4, 8, 4),
        # ("LID-DS-2019","CVE-2017-7529", 4, 4, 8, 16),
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 8, 16),
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 32, 16),
        # ("LID-DS-2019","PHP_CWE-434", 2, 2, 8, 16),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 2, 2, 8, 4),
        ("LID-DS-2019", "SQL_Injection_CWE-89", 2, 2, 32, 4),
        ("LID-DS-2021", "PHP_CWE-434", 2, 2, 32, 16),
        ("LID-DS-2021", "Bruteforce_CWE-307", 2, 2, 8, 16),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 4, 4, 32, 4),
        ("LID-DS-2021", "CVE-2017-7529", 2, 2, 8, 16),
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 32, 4),
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 32, 16),
        # ("LID-DS-2021","CWE-89-SQL-injection", 2, 2, 8, 4),
        ("LID-DS-2021", "CWE-89-SQL-injection", 2, 2, 32, 4),
        ("LID-DS-2021", "CVE-2014-0160", 4, 4, 32, 4),
        # ("LID-DS-2021","CWE-89-SQL-injection", 4, 4, 32, 4),
    ]

    title = f"BOTH"
    file_suffix = ""
    file_path = "transformer/epochs_pre_dedup/"
    file_prefix = f"picked_pre_dedup_rest"
    _picked_rest("final_predup", configs_dict, title, file_path, file_prefix, file_suffix)


def _picked_repeated_rest(collection_name,
                          configs: list[tuple],
                          title,
                          file_path,
                          file_prefix,
                          file_suffix="",
                          NUM_ROWS=2,
                          scale_ths=False):
    all_metrics = {}
    if Path(f"{file_path}/picked_metrics.pkl").exists():
        with open(f"{file_path}/picked_metrics.pkl", "rb") as f:
            print("Loading metrics from cache")
            all_metrics = pickle.load(f)

    query = ResultQuery(collection_name)
    fig, axs = pplt.subplots(
        top=0.5,
        right=2.5,
        left=2.5,
        bottom=0.5,
        hspace=0.5,
        nrows=NUM_ROWS,
        ncols=3,
        refwidth=2.7,
        refheight=2,
        sharey=True,
        sharex=True,
        innerpad=0.1,
    )
    max_fp = 1
    for idx, (dataset, scenario, heads, layers, dim, ngram, run) in enumerate(configs):
        where = {
            "dataset": dataset,
            "scenario": scenario,
            "layers": layers,
            "heads": heads,
            "dim": dim,
            "ng": ngram,
            "run": run,
        }

        results = query.find_results(
            algorithms=["Transformer"],
            scenarios=[scenario],
            datasets=[dataset],
            features=features,
            config_aliases=config_aliases,
            where=where,
        )

        if len(results) == 0:
            print("No results found for", where)
            continue

        if (dataset, scenario, heads, layers, dim, ngram, run) in all_metrics:
            metrics = all_metrics[(dataset, scenario, heads, layers, dim, ngram, run)]
        else:
            metrics = _get_metrics(results, scale_ths)
            all_metrics[(dataset, scenario, heads, layers, dim, ngram, run)] = metrics

        detection_rates, false_positives, ths, epochs_x, train_losses, val_losses = metrics
        idx2 = idx
        axs[idx2].format(
            title=f"{dataset_short[dataset]}-{scenario_short[scenario]} h{heads} l{layers} m{dim} n{ngram}",
            titleloc='lc'
        )

        ln1 = axs[idx2].plot(epochs_x, detection_rates, **DR)
        ln11 = axs[idx2].plot(epochs_x, ths, **THRESHOLD)
        axs[idx2].set_xlim([0, max(epochs_x)])
        axs[idx2].set_ylim([0, max(ths) * 1.05])
        axs[idx2].set_yticks([0, max(ths)])
        axs[idx2].yaxis.grid(which="both")
        axs[idx2].set_ylabel("DR / THR")
        ax12 = axs[idx2].twinx()
        ln12 = ax12.plot(epochs_x, false_positives, **FP)
        max_fp = max(max(false_positives), max_fp)
        ax12.set_ylim([0, max(max(false_positives), 1) * 1.05])
        if (idx + 1) % 3 == 0:
            ax12.set_ylim([0, max_fp * 1.05])
            ax12.set_yticks([0, max_fp])
            ax12.set_ylabel("CFA")
            max_fp = 1
        else:
            ax12.set_yticks([])
        ax12.format(ylabelpad=-6, ylabelsize=12)

    axs[-1, :].format(xlabel="Epochs")
    axs.format(ylabelsize=12, xlabelsize=12)
    axs.format(
        abc='A.',
        abcloc='ul',
        abcweight='bold',
    )
    fig.legend(
        [ln1[0], ln11[0], ln12[0]],
        labels=["Detection Rate", "Threshold", "CFA"],
        loc='b',
        pad=1,
        ncol=5,
        fontsize=20,
        frame=False,
    )

    file_path = f"{file_path}/"
    file_prefix = f"{file_prefix}"
    path = f"{file_path}{file_prefix}_{title.replace(' ', '_')}{file_suffix}.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    pplt.show()
    fig.savefig(path, dpi=400, bbox_inches='tight')

    # cache metrics
    with open(f"{file_path}/picked_metrics.pkl", "wb") as f:
        pickle.dump(all_metrics, f)


def pre_dedup_picked_repeated_rest():
    configs_dict = [
        ("LID-DS-2021", "CVE-2017-7529", 4, 4, 16, 8, 0),
        ("LID-DS-2021", "CVE-2017-7529", 4, 4, 16, 8, 1),
        ("LID-DS-2021", "CVE-2017-7529", 4, 4, 16, 8, 2)
    ]

    title = ""
    file_suffix = ""
    file_path = "transformer/epochs_pre_dedup/"
    file_prefix = f"picked_pre_dedup_repeated_rest"
    _picked_repeated_rest("final_predup", configs_dict, title, file_path, file_prefix, file_suffix, NUM_ROWS=1)


def dedup_picked_with_loss():
    configs_dict = [
        # ("LID-DS-2019","CVE-2017-7529", 2, 2, 32, 16),
        # ("LID-DS-2019","CVE-2017-7529", 4, 4, 8, 4),
        ("LID-DS-2019", "CVE-2017-7529", 4, 4, 8, 16),
        # ("LID-DS-2019","CVE-2014-0160", 4, 4, 8, 16),
        ("LID-DS-2019", "CVE-2012-2122", 2, 2, 8, 16),
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 32, 16),
        # ("LID-DS-2019","PHP_CWE-434", 2, 2, 8, 16),
        # ("LID-DS-2019","PHP_CWE-434", 4, 4, 8, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 2, 2, 8, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 2, 2, 32, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 4, 4, 8, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 4, 4, 32, 4),
        # ("LID-DS-2021","CVE-2017-7529", 2, 2, 8, 16),
        # ("LID-DS-2021","CVE-2014-0160", 4, 4, 32, 4),
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 32, 4),
        ("LID-DS-2021", "Bruteforce_CWE-307", 4, 4, 32, 16),
        # ("LID-DS-2021","PHP_CWE-434", 2, 2, 8, 16),
        # ("LID-DS-2021","CWE-89-SQL-injection", 2, 2, 8, 4),
        # ("LID-DS-2021","CWE-89-SQL-injection", 2, 2, 32, 4),
        # ("LID-DS-2021","CWE-89-SQL-injection", 4, 4, 32, 4),
    ]

    title = "BOTH"
    file_suffix = ""
    file_path = "transformer/epochs_dedup/"
    file_prefix = f"picked_dedup_with_loss"
    scale_ths = True
    _picked(
        "final_dedup",
        configs_dict,
        title,
        file_path,
        file_prefix,
        file_suffix,
        scale_ths=scale_ths
    )


def dedup_picked_rest():
    configs_dict = [
        # ("LID-DS-2019","CVE-2017-7529", 2, 2, 32, 16),
        # ("LID-DS-2019","CVE-2017-7529", 4, 4, 8, 4),
        # ("LID-DS-2019","CVE-2017-7529", 4, 4, 8, 16),
        # ("LID-DS-2019","CVE-2014-0160", 4, 4, 8, 16),
        # ("LID-DS-2019","CVE-2012-2122", 2, 2, 8, 16),
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 32, 16),
        ("LID-DS-2019", "PHP_CWE-434", 2, 2, 8, 16),
        # ("LID-DS-2019","PHP_CWE-434", 4, 4, 8, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 2, 2, 8, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 2, 2, 32, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 4, 4, 8, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 4, 4, 32, 4),
        ("LID-DS-2021", "CVE-2017-7529", 2, 2, 8, 16),
        # ("LID-DS-2021","CVE-2014-0160", 4, 4, 32, 4),
        ("LID-DS-2021", "Bruteforce_CWE-307", 4, 4, 32, 4),
        ("LID-DS-2019", "CVE-2012-2122", 2, 2, 8, 4),
        ("LID-DS-2021", "PHP_CWE-434", 2, 2, 8, 16),
        # ("LID-DS-2021","CWE-89-SQL-injection", 2, 2, 8, 4),
        # ("LID-DS-2021","CWE-89-SQL-injection", 2, 2, 32, 4),
        ("LID-DS-2021", "CWE-89-SQL-injection", 4, 4, 32, 4),
        # ("LID-DS-2021","Bruteforce_CWE-307", 2, 2, 32, 16),
        # ("LID-DS-2019","Bruteforce_CWE-307", 4, 4, 8, 4),
    ]

    title = "BOTH"
    file_suffix = ""
    file_path = "transformer/epochs_dedup/"
    file_prefix = f"picked_dedup_rest"
    scale_ths = True
    _picked_rest("final_dedup", configs_dict, title, file_path, file_prefix, file_suffix, scale_ths=scale_ths)


if __name__ == "__main__":
    pre_dedup_picked_with_loss()
    # pre_dedup_picked_rest()
    # dedup_picked_with_loss()
    # dedup_picked_rest()
    # pre_dedup_picked_repeated_rest()
    # ae_mlp_epoch_analysis_picked()
