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

import argparse
import os

import torch
from dataloader.direction import Direction

from evaluation.fluctuation_analysis.utils import prepare_tf_ngs, ngram_sets, train_tf_model, \
    get_anomaly_scores_for_epochs, cache_losses

learning_rate = 0.001
direction = Direction.OPEN
batch_size = 256


def parser():
    parser = argparse.ArgumentParser(
        description='Fluctuation Analysis with Transformer\n '
                    'Prepare data, train model, and cache anomaly scores and losses.'
    )

    parser.add_argument(
        '-bp', dest='base_path', action='store', type=str, required=True,
        help='LID-DS base path'
    )
    parser.add_argument(
        '-ds', dest='dataset', action='store', type=str, required=True,
        help='LID-DS version'
    )
    parser.add_argument(
        '-s', dest='scenario', action='store', type=str, required=True,
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
        '-e', dest='evaluate', type=lambda x: (str(x).lower() == 'true'), required=True,
        help='Evaluate'
    )

    parser.add_argument(
        '-m', dest='model_dim', action='store', type=int, required=False,
        help='TF model dimension (aka. emb size)'
    )
    parser.add_argument(
        '-l', dest='layers', action='store', type=int, required=False,
        help='Number of encoder and decoder layers'
    )

    parser.add_argument(
        '-nh', dest='num_heads', action='store', type=int, required=False,
        help='Number of model heads'
    )
    parser.add_argument(
        '-cs', dest='custom_split', type=lambda x: (str(x).lower() == 'true'), required=False,
        help='Use custom split'
    )
    parser.add_argument(
        '-do', dest='dropout', action='store', type=float, required=False,
        help='Dropout'
    )
    parser.add_argument(
        '-run', dest='run', action='store', type=float, required=False,
        help='Repeated run number'
    )
    return parser


def main():
    args = parser().parse_args()
    print(args)
    dataset_base = args.base_path
    dataset = args.dataset
    scenario = args.scenario
    checkpoint_dir = args.checkpoint_dir
    ngram_length = args.ngram_length
    evaluate = args.evaluate

    scenario_ngs = prepare_tf_ngs(dataset_base, ngram_length, direction, dataset, scenario, base_path=checkpoint_dir)

    if evaluate:
        model_dim = args.model_dim
        layers = args.layers
        num_heads = args.num_heads
        custom_split = args.custom_split
        dropout = args.dropout
        run = args.run
        if not torch.cuda.is_available():
            print(
                f"CUDA NOT AVAILABLE redo: "
                f"[TF, '{dataset}', '{scenario}', {ngram_length}, {model_dim}, {layers}, {num_heads}, {dropout}, {custom_split}] "
            )
            exit(1)

        NGS = ngram_sets(scenario_ngs)
        syscall_dict, _ = scenario_ngs.syscall_dict

        if run == 0:
            torch.manual_seed(42)
        else:
            checkpoint_dir = os.path.join(checkpoint_dir, f"run_{run}")

        seed_used = torch.initial_seed()
        print(f"seed used: {seed_used}")
        model_tf = train_tf_model(
            scenario,
            dataset,
            ngram_length,
            dropout,
            learning_rate,
            direction,
            custom_split,
            model_dim,
            batch_size,
            syscall_dict,
            num_heads=num_heads,
            layers=layers,
            NGS=NGS,
            epochs=900,
            base_path=checkpoint_dir
        )
        model_tf.use_cache = True
        config = ("TF", dataset, scenario, ngram_length, model_dim, layers, num_heads, dropout, custom_split)
        _ = get_anomaly_scores_for_epochs(model_tf, range(1, 900, 5), NGS, scenario_ngs, config, checkpoint_dir)
        cache_losses(model_tf, config, base_path=checkpoint_dir)


if __name__ == '__main__':
    main()
