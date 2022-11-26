"""
Demonstrates the usage of the result query
"""
from tabulate import tabulate

from algorithms.evaluation.experiment_result_queries import ResultQuery
from dataloader.direction import Direction

features = {
    "Som": ["MaxScoreThreshold", "Som", "Concat", "Ngram", "IntEmbedding"],
    "LSTM": ["MaxScoreThreshold", "LSTM", "Ngram", "W2VEmbedding"]
}

group_by_config = {
    "Som": {
        "name": "MaxScoreThreshold",
        "input": [
            {
                "name": "Som",
                "epochs": "som_epochs",
                "sigma": "sigma",
                "size": "som_size",
                "input": [
                    {
                        "name": "Concat",
                        "input": [
                            {
                                "name": "Ngram",
                                "thread_aware": "thread_aware",
                                "ngram_length": "concat_1_ng_len",
                            },
                            {
                                "name": "Ngram",
                                "ngram_length": "concat_2_ng_len",
                                "input": [
                                    {
                                        "name": "W2VEmbedding",
                                        "vector_size": "concat_2_w2v_size",
                                    },
                                    {
                                        "name": "ReturnValue",
                                        "min_max_scaling": "rv_minmax"
                                    }
                                ]
                            }
                        ]
                    }
                ],
            }
        ],
    },
    "LSTM": {
        "name": "MaxScoreThreshold",
        "input": [
            {
                "name": "LSTM",
                "batch_size": "lstm_batch_size",
                "epochs": "epochs",
                "input": [
                    {
                        "name": "Ngram",
                        "thread_aware": "thread_aware",
                    }
                ],
            },
        ],
    }
}


def find_best_algorithm():
    """
    Finds the best algorithm given some features sorted by average DR
    """

    results = ResultQuery(collection_name="experiments_test").find_best_algorithm(
        algorithms=["Som", "LSTM"],
        scenarios=["CVE-2017-7529", "CVE-2014-0160"],
        directions=[Direction.BOTH],
        features=features,
        group_by_config=group_by_config,
    )

    print(tabulate(results, headers="keys", tablefmt="github"))


def algorithm_wise_best_average_configuration():
    """
    Algorithm wise best average configuration over given scenario and features
    """
    results = ResultQuery(collection_name="experiments_test").algorithm_wise_best_average(
        algorithms=["Som", "LSTM"],
        scenarios=["CVE-2017-7529", "CVE-2014-0160"],
        directions=[Direction.BOTH],
        features=features,
        group_by_config=group_by_config,
        firstK_in_group=3
    )

    for result in results:
        print(result['_id'])

        r = [r.pop("_id") | r for r in result['results']]

        print(tabulate(r, headers="keys", tablefmt="github"))


if __name__ == '__main__':
    print("########## best algorithm ##########")
    find_best_algorithm()

    print("########## algo wise best configuration ##########")
    algorithm_wise_best_average_configuration()
