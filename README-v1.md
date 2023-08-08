# The Room environment - v1

[![PyPI version](https://badge.fury.io/py/room-env.svg)](https://badge.fury.io/py/room-env)

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

env = gym.make("room_env:RoomEnv-v1")
(observation, question), info = env.reset()
rewards = 0

while True:
    # There is one different thing in the RoomEnv from the original AAAI-2023 paper:
    # The reward is either +1 or -1, instead of +1 or 0.
    (observation, question), reward, done, truncated, info = env.step("This is my answer!")
    rewards += reward
    if done:
        break
```

```json
{
    "des_size": "l",
    "seed": 42,
    "question_prob": 1.0,
    "allow_random_human": False,
    "allow_random_question": False,
    "total_episode_rewards": 128,
    "check_resources": True,
}
```

If you want to create an env with a different set of parameters, you can do so. For example:

```python
env_params = {"seed": 0,
              "allow_random_human": True,
              "pretrain_semantic": True}
env = gym.make("room_env:RoomEnv-v1", **env_params)
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

## [Cite our paper](https://doi.org/10.1609/aaai.v37i1.25075)

```bibtex
@article{Kim_Cochez_Francois-Lavet_Neerincx_Vossen_2023,
  title={A Machine with Short-Term, Episodic, and Semantic Memory Systems}, volume={37},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/25075},
  DOI={10.1609/aaai.v37i1.25075},
  abstractNote={Inspired by the cognitive science theory of the explicit human memory systems, we have modeled an agent with short-term, episodic, and semantic memory systems, each of which is modeled with a knowledge graph. To evaluate this system and analyze the behavior of this agent, we designed and released our own reinforcement learning agent environment, “the Room”, where an agent has to learn how to encode, store, and retrieve memories to maximize its return by answering questions. We show that our deep Q-learning based agent successfully learns whether a short-term memory should be forgotten, or rather be stored in the episodic or semantic memory systems. Our experiments indicate that an agent with human-like memory systems can outperform an agent without this memory structure in the environment.},
  number={1},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Kim, Taewoon and Cochez, Michael and Francois-Lavet, Vincent and Neerincx, Mark and Vossen, Piek},
  year={2023},
  month={Jun.},
  pages={48-56}
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
