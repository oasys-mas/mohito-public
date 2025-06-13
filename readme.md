# 🍹 MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for Task-Open Systems

This is the implementation for `MOHITO` our multi-agent reinforcement learning (MARL) solution for task-open Markov games (TaO-MG). This work is published in the paper: [MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for Task-Open Systems]() at UAI 2025. 

**Wildfire** and **Rideshare**, our two domains, have different requirements. **Wildfire** is a fork of [Free-Range-Zoo](), an api for open MARL environments. This fork allows for multiple wildfires to exist in the same cell.  **Rideshare** is a precursor to the free-range-zoo rideshare and operates slightly differently.

Below are the instructions to install and run Rideshare and Wildfire. They have different requirements, so be careful to avoid dependency conflicts. 

---

## Rideshare




## Wildfire

### Installation

1. Clone the free-range-zoo submodule `git submodule update --init`
2. Install [nested-wildfire-free-range-zoo](./wildfire/nested-wildfire-free-range-zoo/), using the free-range-zoo [installation instructions](https://oasys-mas.github.io/free-range-zoo/introduction/installation.html) 
3. Install [MOHITO](./wildfire/mohito/) with `poetry install` from the `wildfire/mohito` directory.
4. Install `scipy`, `seaborn`, `tqdm`, and `numpy` for plotting. `pip install scipy seaborn numpy`.


### [Training](./wildfire/mohito_wf_trainer.py)
1. `-ol` select an openness level. This determines the rate of fire spread.
2. `-m` Select a configuration (e.g. [wf_frank_no_drop.yaml](./wildfire/configs/wf_frank_no_drop.yaml)). This controls the settings for training, testing, and MOHITO hyperparameters.
3. `-o` Select an output path (e.g. `results/test_A`). 

---

### [Testing](./wildfire/mohito_wf_tester.py)

1. `-ol` selects an openness level. This determines the rate of fire spread. In our paper, this matches the training ol.
2. `-m` selects a configuration this must match the one in training.
3. `-o` selects a output directory. The logs for testing will be placed here split by `starting_state`, `checkpoint`, and `openness_level`
4. `-p` is the policy path. This should be `-o/mohito_checkpoints` from training.
5. `--seed` is the seed used for testing. This will be the initial seed for the first starting state. Reproducibility is dependent on this and `n`.
6. `-n` is the number of episodes to perform per openness level. This is split evenly across openness levels. 
7. (Optional) `--low` is the starting index of checkpoints to test.
8. (Optional) `--high` is the ending index of checkpoints to test (-1 is all).


---

### [Baselines](./wildfire/baseline_generator.py)

1. `-o` the output folder for baselines. 
2. `--seed` same as testing.
3. `-n` same as testing.

---

### [Plotting](./wildfire/plotting_wildfire.ipynb)

1. Set `policy_eval_outputs` values to policy root paths. 
2. Set `baseline_output_folder` to output folder for Baselines.
3. Run the notebook.



## Citation

If you use this work please cite us with this,

```
@inproceedings{mohito,
    author = { Anil, Gayathri and Doshi, Prashant and Redder, Daniel and Eck, Adam and Soh, Leen-Kiat},
    booktitle = {UAI},
    title = {MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for Task-Open Systems},
    year = {2025}
}
```