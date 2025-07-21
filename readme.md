# 🍹 MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for Task-Open Systems

This is the implementation for `MOHITO` our multi-agent reinforcement learning (MARL) solution for task-open Markov games (TaO-MG). This work is published in the paper: [MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for Task-Open Systems]() at UAI 2025. 

**Wildfire** and **Rideshare**, our two domains, have different requirements. **Wildfire** is a fork of [Free-Range-Zoo](https://github.com/oasys-mas/free-range-zoo), an api for open MARL environments. This fork allows for multiple wildfires to exist in the same cell.  **Rideshare** is a precursor to the free-range-zoo rideshare and operates slightly differently.

Below are the instructions to install and run Rideshare and Wildfire. They have different requirements, so be careful to avoid dependency conflicts. 

---

## Rideshare

### Training
- Training is done using the `trainer.py` script.
- All required parameters including number of agents, openness level, pooling limit, simulated episode type, and training hyperparameters can be set within the script.
- Trained policy files are saved to `./results/<run_name>/model_files/` by default, where `<run_name>` is automatically generated based on experiment settings.
- The following files are also saved under the same folder:
  - `policy_agent<i>.pth` and `critic_agent<i>.pth`: Checkpoints for each agent.
  - `losses_<run_name>.pkl`: Contains loss metrics for actor and critic during training.
  - `eval_file_<run_name>.csv`: Evaluation logs captured at periodic intervals.
  - `stats_file_<run_name>.csv`: Per-step environment statistics (e.g., trip success, wait times).
  - `eval_stats_file_<run_name>.csv`: Evaluation-time environment statistics.
  - `config.json`: Dumped hyperparameters and training configuration.
  - `random_actor.pth`: A random baseline actor for comparison.
- Post-training analysis (e.g., plotting losses) is performed via `loss_analysis_split.py`.

### Testing
- Testing is done using the `evaluator.py` script.
- Set the number of agents and the policy files you would like to use to test a scenario.
- You can run this on a subset of policies (just OL1+2agent or all. If you do it individually concatenate the `MOHITO` portions into one csv before plotting)
- Environment variables such as openness level, grid dimensions, etc. can be set in the script.
- During evaluation, the following outputs are saved:
  - A `.csv` file with detailed step-wise agent performance, stored under `./baseline-comparisons/`
  - An optional `temp.pkl` file containing full trajectory data for visualization (saved under `./results/<result_folder>/` if enabled in code)

- follow the instructions in the [TaO-PG-ELLA Readme](./rideshare/tao_pg_ella/README.md) for our remaining baseline.

### [Plotting](./rideshare/plotting_rideshare.ipynb)
- Set the `result_file` as the path to the evaluator output after running across all openness levels and agent numbers. 
- Plotting can be run on a subset, but some plotting cells may throw errors. 


## Wildfire

### Installation

1. Clone the free-range-zoo submodule `git submodule update --init`
2. Install [nested-wildfire-free-range-zoo](https://github.com/oasys-mas/nested-wildfire-free-range-zoo), using the free-range-zoo [installation instructions](https://oasys-mas.github.io/free-range-zoo/introduction/installation.html) 
3. Install [MOHITO](./wildfire/mohito/) with `poetry install` from the `wildfire/mohito` directory.
4. Install `scipy`, `seaborn`, `tqdm`, and `numpy` for plotting. `pip install scipy seaborn numpy tqdm`.


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
