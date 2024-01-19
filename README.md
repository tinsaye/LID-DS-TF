# LID-DS Transformer

Source code for the implementation and evaluation of a Transformer-based Anomaly Detection on Streams of System Calls (Master’s Thesis)

The thesis aimed to develop a host-based intrusion detection system (HIDS) using
a transformer-based model as an anomaly detector. The proposed system builds a model of
normal behaviour by processing n-grams of system calls during training. The trained model
is then used to detect anomalies when an attack occurs by measuring the deviation from the
benign profile.
The architecture of the transformer-based model consists of stacked decoders to allow a
language modelling approach while processing the n-grams of system calls. The development
and evaluation of the HIDS employed the [Leipzig Intrusion Detection – Data Set (LID-DS)](https://github.com/LID-DS/LID-DS) and
framework.

## Project Structure

```text
src
├── cluster                     # scripts for running the ids pipline on an HPC cluster with slurm
├── decision_engines            # transformer and a modified AE decision engine, also contains the transformer model
├── features                    # several building blocks that could be used as input features for the model.
├── evaluation                  # script and notebooks used for creating the evaluation plots and tables
│   ├── fluctuation_analysis    # utility classes and script for creating the ngram set experiments (sections 5.3 and 6.3 of my thesis)
│   │   └── cluster             # scripts to run these analysis on cluster
│   ├── js_functions            # MongoDB custom js functions used to retrieve/aggregate saved results (lid_ds specifics)
│   ├── preliminary             # scripts used to create the plots for the preliminary experiments (6.2)
│   └── primary                 # plots for the final full evaluation (6.4)
├── misc                        # scripts that can be used to run the AE and MLP based IDSs
├── Models                      # empty directory used as checkpoint for trained models
└── utils                       # helper functions to store and load trained models for a specific epochs

```
> [!NOTE]
> There are several features in the `src/features` directory that did not make it into my thesis due to limited project scope and changes in research direction.

## Requirements

- Python 3.9
- [LID-DS](https://github.com/LID-DS/LID-DS)
- pytorch (1.9.0)
- Dataset: see [LID-DS](https://github.com/LID-DS/LID-DS) readme

## Installation

1. Set up your python environment (venv, ...)

2. Install main dependency LID-DS from source
```shell
cd /path/for/installing/lidds
git clone git@github.com:LID-DS/LID-DS.git
cd LID-DS
git checkout 0f7760a4785f8758359227a1309be46b5d14955a   # to ensure compatibility, last commit at the time of development
pip install -r requirements.txt                         # this is important as the setup script does not install all dependencies
pip install -e .                                        # install LID-DS from source
```

3. Install other dependencies
```shell
cd /path/to/this/project/LID-DS-TF
pip install -r requirements.txt
```

## Usage

To run the IDS pipeline on the LID-DS dataset for a single scenario with default configurations, use the following command:
```shell
cd src
LID_DS_BASE="/path/to/Datasets/" python ids_transformer_main.py
```
The default configuration can be found in the main function of `ids_transformer_main.py`.

