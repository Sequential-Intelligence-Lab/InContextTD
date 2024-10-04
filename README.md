# InContextTD

Welcome to the InContextTD repository, which is the repo for the paper [Transformers Learn Temporal Difference Methods for in Context Reinforcement Learning](https://arxiv.org/abs/2405.13861)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction
With this repository, 

## Installation
To install InContextTD, follow these steps:

1. Clone the repository:
2. Navigate to the project directory
3. Install the required dependencies:
    ```bash
    python setup.py
    ```

## Usage
To start using InContextTD, run the following command:

To run a single experiment run

```bash
python main.py [options]
```

Here is the complete list command line arguments you can use with `main.py`:

- `--linear`: Specify whether to train a linear or nonlinear transformer (flag).
- `-d`, `--dim_feature`: Feature dimension (default: 4).
- `-s`, `--num_states`: Number of states (default: 10).
- `-n`, `--context_length`: Context length (default: 30).
- `-l`, `--num_layers`: Number of layers (default: 3).
- `--gamma`: Discount factor (default: 0.9).
- `--lmbd`: Eligibility trace decay rate (default: 0.0).
- `--activation`: Activation function for the transformer (choices: ['softmax','relu','identity']).
- `--sample_weight`: Sample a random true weight vector, such that the value function is fully representable by the features (flag).
- `--n_mrps`: Total number of MRPs for training (default: 4,000).
- `--batch_size`: Mini batch size (default: 64).
- `--n_batch_per_mrp`: Number of mini-batches sampled from each MRP (default: 5).
- `--lr`: Learning rate (default: 0.001).
- `--weight_decay`: Regularization term (default: 1e-6).
- `--log_interval`: Logging interval (default: 10).
- `--mode`: Training mode: auto-regressive or sequential (default: 'auto', choices: ['auto', 'sequential']).
- `--seed`: Random seed (default: list(range(1,30))).
- `--save_dir`: Directory to save logs (default: None).
- `--suffix`: Suffix to add to the save directory (default: None).
- `--gen_gif`: Generate a GIF for the evolution of weights (flag).
- `-v`, `--verbose`: Print training details (flag).

By default, the runs will save in `./logs/YYYY-MM-DD-HH-MM-SS` format.

For example, to replicate Figure 1 in the paper, run
```bash
python main.py python main.py --suffix=linear_standard --l=3 --activation=identity --mode=auto -v
```
and the figures averaged across all the seeds will be saved in `./logs/YYYY-MM-DD-HH-MM-SS/linear_standard/averaged_figures`, with diagnostic figures for each seed stored in `./logs/YYYY-MM-DD-HH-MM-SS/linear_standard/seed_SEED/figures`.

It will yield the following plots
![P Metrics Plot](figs/P_metrics_1.pdf)
![Q Metrics Plot](figs/Q_metrics_1.pdf)


To run all of the experiments from the paper, simply run the shell script
```bash
./run.sh
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.