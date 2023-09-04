# The Room environment - v1

[![PyPI version](https://badge.fury.io/py/room-env.svg)](https://badge.fury.io/py/room-env)

We have released a challenging [Gymnasium](https://www.gymlibrary.dev/) compatible
environment.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.8 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.
1. This env is added to the PyPI server. Just run: `pip install room-env`

## RoomEnv-v2

```python
import gymnasium as gym
import random

room_config = {
    "officeroom": {
        "north": "wall",
        "east": "livingroom",
        "south": "wall",
        "west": "wall",
    },
    "livingroom": {
        "north": "wall",
        "east": "wall",
        "south": "bedroom",
        "west": "officeroom",
    },
    "bedroom": {
        "north": "livingroom",
        "east": "wall",
        "south": "wall",
        "west": "wall",
    },
}


object_transition_config = {
    "static": {"bed": None, "desk": None, "table": None},
    "independent": {
        "tae": {
            "officeroom": {"north": 0, "east": 0.1, "south": 0, "west": 0, "stay": 0.9},
            "livingroom": {
                "north": 0,
                "east": 0,
                "south": 0,
                "west": 0.1,
                "stay": 0.9,
            },
            "bedroom": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
        },
        "michael": {
            "officeroom": {
                "north": 0,
                "east": 0,
                "south": 0,
                "west": 0,
                "stay": 0,
            },
            "livingroom": {
                "north": 0,
                "east": 0,
                "south": 0.9,
                "west": 0,
                "stay": 0.1,
            },
            "bedroom": {"north": 0.1, "east": 0, "south": 0, "west": 0, "stay": 0.9},
        },
        "vincent": {
            "officeroom": {
                "north": 0,
                "east": 0.5,
                "south": 0,
                "west": 0,
                "stay": 0.5,
            },
            "livingroom": {
                "north": 0,
                "east": 0,
                "south": 0.333,
                "west": 0.333,
                "stay": 0.333,
            },
            "bedroom": {
                "north": 0.5,
                "east": 0,
                "south": 0,
                "west": 0,
                "stay": 0.5,
            },
        },
    },
    "dependent": {
        "laptop": {"tae": 0.7, "michael": 0.4, "vincent": 0.1},
        "phone": {"tae": 0.1, "michael": 0.7, "vincent": 0.4},
        "headset": {"tae": 0.4, "michael": 0.1, "vincent": 0.9},
    },
    "agent": {
        "agent": {"officeroom": None, "livingroom": None, "bedroom": None},
    },
}

object_init_config = {
    "static": {
        "bed": {"officeroom": 0, "livingroom": 0, "bedroom": 1},
        "desk": {"officeroom": 1, "livingroom": 0, "bedroom": 0},
        "table": {"officeroom": 0, "livingroom": 1, "bedroom": 0},
    },
    "independent": {
        "tae": {"officeroom": 0.5, "livingroom": 0.5, "bedroom": 0},
        "michael": {"officeroom": 0, "livingroom": 0.5, "bedroom": 0.5},
        "vincent": {"officeroom": 0.333, "livingroom": 0.333, "bedroom": 0.333},
    },
    "dependent": {
        "laptop": {"officeroom": 0.333, "livingroom": 0.333, "bedroom": 0.333},
        "phone": {"officeroom": 0.333, "livingroom": 0.333, "bedroom": 0.333},
        "headset": {"officeroom": 0.333, "livingroom": 0.333, "bedroom": 0.333},
    },
    "agent": {
        "agent": {"officeroom": 0.333, "livingroom": 0.333, "bedroom": 0.333},
    },
}

config = {
    "room_config": room_config,
    "object_transition_config": object_transition_config,
    "object_init_config": object_init_config,
    "question_prob": 1.0,
    "seed": 42,
    "terminates_at": 99,
}


env = gym.make("room_env:RoomEnv-v2", **config)
(observation, question), info = env.reset()
rewards = 0

while True:
    action_qa = question[0]
    action_explore = random.choice(["north", "east", "south", "west", "stay"])
    (obs, question), reward, done, truncated, info = env.step(("wall", action_explore))
    rewards += reward
    if done:
        break


```

Take a look at [this repo](https://github.com/tae898/explicit-memory) for an actual
interaction with this environment to learn a policy.

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

## [Cite our paper](WIP)

```bibtex
WIP
```

## Authors

- [Taewoon Kim](https://taewoon.kim/)
- [Michael Cochez](https://www.cochez.nl/)
- [Vincent Francois-Lavet](http://vincent.francois-l.be/)

## License

[MIT](https://choosealicense.com/licenses/mit/)
