# The Room environment - v1
[![PyPI version](https://badge.fury.io/py/room-env.svg)](https://badge.fury.io/py/room-env)

For the documentation of [RoomEnv-v0](./documents/README-v0.md), click the corresponding buttons.

This document, RoomEnv-v1, is the most up-to-date one.

We have released a challenging [Gymnasium](https://www.gymlibrary.dev/) compatible
environment. The best strategy for this environment is to have both episodic and semantic
memory systems. See the [paper](https://arxiv.org/abs/2212.02098) for more information.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.8 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.
1. This env is added to the PyPI server. Just run: `pip install room-env`

## RoomEnv-v1

```python
import gymnasium as gym
import room_env
import random

env = gym.make("RoomEnv-v1")
observation, info = env.reset()
rewards = 0

while True:
    # There is one different thing in the RoomEnv from the original AAAI-2023 paper:
    # The reward is either +1 or -1, instead of +1 or 0.
    observation, reward, done, truncated, info = env.step(random.randint(0, 2))
    rewards += reward
    if done:
        break

print(rewards)
```

Every time when an agent takes an action, the environment will give you three memory
systems (i.e., episodic, semantic, and short-term), as an `observation`. The goal of the
agent is to learn a memory management policy. The actions are:

- 0: Put the short-term memory into the episodic memory system.
- 1: Put it into the semantic.
- 2: Just forget it.

The memory systems will be managed according to your actions, and they will eventually
be used to answer questions. You don't have to worry about the question answering. It's done
by the environment. The better you manage your memory systems, the higher chances that
your agent can answer more questions correctly!

The default parameters for the environment are

```json
{
    "des_size": "l",
    "seed": 42,
    "policies": {"encoding": "argmax",
                "memory_management": "RL",
                "question_answer": "episodic_semantic"},
    "capacity": {"episodic": 16, "semantic": 16, "short": 1},
    "question_prob": 1.0,
    "observation_params": "perfect",
    "allow_random_human": False,
    "allow_random_question": False,
    "total_episode_rewards": 128,
    "pretrain_semantic": False,
    "check_resources": True,
    "varying_rewards": False
}
```

If you want to create an env with a different set of parameters, you can do so. For example:

```python
env_params = {"seed": 0,
              "capacity": {"episodic": 8, "semantic": 16, "short": 1},
              "pretrain_semantic": True}
env = gym.make("RoomEnv-v1", **env_params)
```

Take a look at [this repo](https://github.com/tae898/explicit-memory) for an actual
interaction with this environment to learn a policy.

## Data collection

Data is collected from querying ConceptNet APIs. For simplicity, we only collect triples
whose format is (`head`, `AtLocation`, `tail`). Here `head` is one of the 80 MS COCO
dataset categories. This was kept in mind so that later on we can use images as well.

If you want to collect the data manually, then run below:

```
python collect_data.py
```

## [The RoomDes](room_env/des.py)

The DES is part of RoomEnv. You don't have to care about how it works. If you are still
curious, you can read below.

You can run the RoomDes by

```python
from room_env.des import RoomDes

des = RoomDes()
des.run(debug=True)
```

with `debug=True` it'll print events (i.e., state changes) to the console.

```console
{'resource_changes': {'desk': -1, 'lap': 1},
 'state_changes': {'Vincent': {'current_time': 1,
                               'object_location': {'current': 'desk',
                                                   'previous': 'lap'}}}}
{'resource_changes': {}, 'state_changes': {}}
{'resource_changes': {}, 'state_changes': {}}
{'resource_changes': {},
 'state_changes': {'Michael': {'current_time': 4,
                               'object_location': {'current': 'lap',
                                                   'previous': 'desk'}},
                   'Tae': {'current_time': 4,
                           'object_location': {'current': 'desk',
                                               'previous': 'lap'}}}}
```

## Contributing

Contributions are what make the open source community such an amazing place to be learn,
inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory,
   to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## [Cite our paper](https://arxiv.org/abs/2212.02098)

```bibtex
@misc{https://doi.org/10.48550/arxiv.2212.02098,
  doi = {10.48550/ARXIV.2212.02098},
  url = {https://arxiv.org/abs/2212.02098},
  author = {Kim, Taewoon and Cochez, Michael and François-Lavet, Vincent and Neerincx, Mark and Vossen, Piek},
  keywords = {Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A Machine with Short-Term, Episodic, and Semantic Memory Systems},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Cite our code

[![DOI](https://zenodo.org/badge/477781069.svg)](https://zenodo.org/badge/latestdoi/477781069)

## Authors

- [Taewoon Kim](https://taewoon.kim/)
- [Michael Cochez](https://www.cochez.nl/)
- [Vincent Francois-Lavet](http://vincent.francois-l.be/)
- [Mark Neerincx](https://ocw.tudelft.nl/teachers/m_a_neerincx/)
- [Piek Vossen](https://vossen.info/)

## License

[MIT](https://choosealicense.com/licenses/mit/)
