import pickle
from pathlib import Path

import proplot as pplt
from algorithms.evaluation.experiment_result_queries import ResultQuery
from matplotlib.ticker import FormatStrFormatter

from evaluation.preliminary.common import TRAIN, VAL, THRESHOLD, DR, FP, config_aliases, features, dataset_short, scenario_short, \
    _get_metrics


# noinspection DuplicatedCode
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
    print(NUM_ROWS)
    query = ResultQuery(collection_name)
    fig, axs = pplt.subplots(
        top=0.5,
        right=2.5,
        left=3,
        bottom=0.5,
        hspace=0.8,
        nrows=NUM_ROWS,
        ncols=3,
        refwidth=2.7,
        refheight=2,
        sharey=False,
        sharex=True,
        innerpad=0.4,
    )
    max_fp = 1
    for idx, (dataset, scenario, heads, layers, dim, ngram) in enumerate(configs):
        where = {
            "dataset": dataset,
            "scenario": scenario,
            "layers": layers,
            "heads": heads,
            "dim": dim,
            "ngram": ngram,
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
        idx1 = (idx // 3 * 2, idx % 3)
        idx2 = (idx // 3 * 2 + 1, idx % 3)
        losses_x = range(1, len(train_losses) + 1)
        print(idx1)
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
        axs[idx1].yaxis.grid(which="both")
        axs[idx1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[idx1].format(
            title=f"{dataset_short[dataset]}-{scenario_short[scenario]} h{heads} l{layers} m{dim} n{ngram}",
            titleloc='ur'
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
        ax12 = axs[idx2].twinx()
        ln12 = ax12.plot(
            epochs_x,
            false_positives,
            **FP,
        )
        max_fp = max(max(false_positives), max_fp)
        ax12.set_ylim([0, max(max(false_positives), 1) * 1.05])
        ax12.set_ylim([0, max_fp * 1.05])
        ax12.set_yticks([0, max_fp])
        if idx % 3 == 2:
            ax12.set_ylim([0, max_fp * 1.05])
            ax12.set_yticks([0, max_fp])
            ax12.set_ylabel("CFA")
        else:
            ax12.set_yticks([])
        ax12.format(
            ylabelpad=-8,
            ylabelsize=12,
        )

        # legend
    axs[-1, :].format(xlabel="Epochs")
    axs[0, 0].format(ylabel="Loss", ylabelpad=-4)
    axs[1, 0].format(ylabel="DR / THR")
    axs[2, 0].format(ylabel="Loss", ylabelpad=-4)
    axs[3, 0].format(ylabel="DR / THR")
    axs[4, 0].format(ylabel="Loss", ylabelpad=-4)
    axs[5, 0].format(ylabel="DR / THR")

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
            "ngram": ngram,
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


# noinspection DuplicatedCode
def pre_dedup_picked_with_loss():
    configs_dict = [
        ## ("LID-DS-2019", "CVE-2017-7529", 4, 4, 8, 4),
        # ("LID-DS-2019","CVE-2017-7529", 2, 2, 8, 4),
        ("LID-DS-2019", "CVE-2017-7529", 4, 4, 8, 16),
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 8, 16),
        ("LID-DS-2019", "CVE-2012-2122", 2, 2, 8, 16),
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 32, 16),
        ("LID-DS-2019", "PHP_CWE-434", 2, 2, 8, 16),
        ##("LID-DS-2019","SQL_Injection_CWE-89", 2, 2, 8, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 2, 2, 32, 4),
        ("LID-DS-2019", "SQL_Injection_CWE-89", 4, 4, 32, 4),
        # ("LID-DS-2021","CVE-2017-7529", 2, 2, 8, 16),
        ("LID-DS-2021", "CVE-2014-0160", 4, 4, 32, 4),
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 32, 4),
        ("LID-DS-2021", "Bruteforce_CWE-307", 4, 4, 32, 16),
        ("LID-DS-2021", "PHP_CWE-434", 2, 2, 32, 16),
        ("LID-DS-2021", "CWE-89-SQL-injection", 2, 2, 8, 4),
        # ("LID-DS-2021","CWE-89-SQL-injection", 2, 2, 32, 4),
        ("LID-DS-2021", "CWE-89-SQL-injection", 4, 4, 32, 4),
    ]

    title = f"BOTH"
    file_suffix = ""
    file_path = "transformer_appendix/epochs_pre_dedup/"
    file_prefix = f"picked_pre_dedup_with_loss_appendix"
    _picked(
        "final_predup",
        configs_dict,
        title,
        file_path,
        file_prefix,
        file_suffix,
        NUM_ROWS=2 * len(configs_dict) // 3
    )


# noinspection DuplicatedCode
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
    file_path = "transformer_appendix/epochs_pre_dedup/"
    file_prefix = f"picked_pre_dedup_rest_appendix"
    _picked_rest("final_predup", configs_dict, title, file_path, file_prefix, file_suffix)


# noinspection DuplicatedCode
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
            "ngram": ngram,
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
    # fig.save(path, dpi=400)
    fig.savefig(path, dpi=400, bbox_inches='tight')
    # pplt.close(fig)

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
    file_path = "transformer_appendix/epochs_pre_dedup/"
    file_prefix = f"picked_pre_dedup_repeated_rest"
    _picked_repeated_rest("final_predup", configs_dict, title, file_path, file_prefix, file_suffix, NUM_ROWS=1)


def dedup_picked_with_loss():
    configs_dict = [
        ## ("LID-DS-2019", "CVE-2017-7529", 4, 4, 8, 4),
        # ("LID-DS-2019","CVE-2017-7529", 2, 2, 8, 4),
        ("LID-DS-2019", "CVE-2017-7529", 4, 4, 8, 16),
        ("LID-DS-2019", "CVE-2012-2122", 4, 4, 8, 16),
        ## ("LID-DS-2019", "CVE-2012-2122", 2, 2, 8, 16),
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 32, 16),
        ("LID-DS-2019", "PHP_CWE-434", 2, 2, 8, 16),
        ("LID-DS-2019", "SQL_Injection_CWE-89", 2, 2, 8, 4),
        # ("LID-DS-2019","SQL_Injection_CWE-89", 2, 2, 32, 4),
        ("LID-DS-2019", "SQL_Injection_CWE-89", 4, 4, 32, 4),
        # ("LID-DS-2021","CVE-2017-7529", 2, 2, 8, 16),
        ("LID-DS-2021", "CVE-2014-0160", 4, 4, 32, 4),
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 32, 4),
        ## ("LID-DS-2021", "Bruteforce_CWE-307", 4, 4, 32, 16),
        ("LID-DS-2021", "PHP_CWE-434", 2, 2, 32, 16),
        ("LID-DS-2021", "CWE-89-SQL-injection", 2, 2, 8, 4),
        # ("LID-DS-2021","CWE-89-SQL-injection", 2, 2, 32, 4),
        ("LID-DS-2021", "CWE-89-SQL-injection", 4, 4, 32, 4),
    ]

    title = "BOTH"
    file_suffix = ""
    file_path = "transformer_appendix/epochs_dedup/"
    file_prefix = f"picked_dedup_with_loss_appendix"
    scale_ths = True
    _picked(
        "final_dedup", configs_dict, title, file_path, file_prefix, file_suffix, scale_ths=scale_ths,
        NUM_ROWS=2 * len(configs_dict) // 3
    )


def dedup_picked_with_loss_problem():
    configs_dict = [
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 32, 16),
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 32, 16),
        # ("LID-DS-2019","CVE-2012-2122", 4, 4, 32, 16),
        # ("LID-DS-2021","CVE-2017-7529", 2, 2, 8, 16),
        # ("LID-DS-2021","CVE-2017-7529", 2, 2, 8, 16),
        # ("LID-DS-2021","CVE-2017-7529", 2, 2, 8, 16),

        ("LID-DS-2019", "CVE-2012-2122", 2, 2, 8, 4),  #
        ("LID-DS-2019", "CVE-2012-2122", 2, 2, 8, 16),  #
        ("LID-DS-2019", "CVE-2012-2122", 2, 2, 32, 4),  #
        # ("LID-DS-2019", "CVE-2012-2122", 2, 2, 32, 16),
        # ("LID-DS-2019", "CVE-2012-2122", 4, 4, 8, 4),
        # ("LID-DS-2019", "CVE-2012-2122", 4, 4, 8, 16),
        # ("LID-DS-2019", "CVE-2012-2122", 4, 4, 32, 4),
        # ("LID-DS-2019", "CVE-2012-2122", 4, 4, 32, 16),
        # ("LID-DS-2019", "CVE-2012-2122", 4, 4, 32, 16),

        # ("LID-DS-2021", "CVE-2017-7529", 2, 2, 8, 4),
        ("LID-DS-2021", "CVE-2017-7529", 2, 2, 8, 16),  #
        # ("LID-DS-2021", "CVE-2017-7529", 2, 2, 32, 4),
        # ("LID-DS-2021", "CVE-2017-7529", 2, 2, 32, 16),
        ("LID-DS-2021", "CVE-2017-7529", 4, 4, 8, 4),  #
        # ("LID-DS-2021", "CVE-2017-7529", 4, 4, 8, 16),
        # ("LID-DS-2021", "CVE-2017-7529", 4, 4, 32, 4),
        # ("LID-DS-2021", "CVE-2017-7529", 4, 4, 32, 16),
        ("LID-DS-2021", "CVE-2017-7529", 4, 4, 32, 16),  #

        ("LID-DS-2019", "Bruteforce_CWE-307", 2, 2, 32, 4),  #
        ("LID-DS-2021", "Bruteforce_CWE-307", 2, 2, 8, 4),  #
        # ("LID-DS-2021","Bruteforce_CWE-307", 2, 2, 8, 16),
        # ("LID-DS-2021","Bruteforce_CWE-307", 2, 2, 32, 4),
        ("LID-DS-2021", "Bruteforce_CWE-307", 2, 2, 32, 16),  #
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 8, 4),
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 8, 16),
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 32, 4),
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 32, 16),
        # ("LID-DS-2021","Bruteforce_CWE-307", 4, 4, 32, 16), #

        ## ("LID-DS-2021", "Bruteforce_CWE-307", 4, 4, 32, 16),
    ]

    title = "BOTH"
    file_suffix = ""
    file_path = "transformer_appendix/epochs_dedup/"
    file_prefix = f"picked_dedup_with_loss_problemo"
    scale_ths = True
    _picked(
        "final_dedup",
        configs_dict,
        title,
        file_path,
        file_prefix,
        file_suffix,
        scale_ths=scale_ths,
        NUM_ROWS=2 * len(configs_dict) // 3
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
    file_path = "transformer_appendix/epochs_dedup/"
    file_prefix = f"picked_dedup_rest"
    scale_ths = True
    _picked_rest("final_dedup", configs_dict, title, file_path, file_prefix, file_suffix, scale_ths=scale_ths)


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
            # ("CVE-2018-3760", 5, 0.5, 0),
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
                "ngram": ngram if algo == "AE" else ngram + 1,
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

    file_path = f"transformer_appendix/epochs_dedup/"
    file_suffix = "_OTHERS"
    file_prefix = "picked_dedup_"
    path = f"{file_path}{file_prefix}{title.replace(' ', '_')}{file_suffix}.png"
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=400, bbox_inches='tight')

    pplt.show()


if __name__ == "__main__":
    pre_dedup_picked_with_loss()
