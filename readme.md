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
3. Install [MOHITO](./wildfire/MOHITO/) with `poetry install` from the `wildfire/mohito` directory.


### [Training](./wildfire/mohito_wf_trainer.py)
1. `-ol` select an openness level. This determines the rate of fire spread.
2. `-m` Select a configuration (e.g. [wf_frank_no_drop.yaml](./wildfire/configs/wf_frank_no_drop.yaml)). This controls the settings for training, testing, and MOHITO hyperparameters.
3. `-o` Select an output path (e.g. `results/test_A`). 

---

### [Testing](./wildfire/mohito_wf_tester.py)

---

### [Baselines](./wildfire/baseline_generator.py)



## Citation

If you use this work cite you can cite us with this,

```
@inproceedings{mohito,
    author = { Anil, Gayathri and Doshi, Prashant and Redder, Daniel and Eck, Adam and Soh, Leen-Kiat},
    booktitle = {UAI},
    title = {MOHITO: Multi-Agent Reinforcement Learning using Hypergraphs for Task-Open Systems},
    year = {2025}
}
```