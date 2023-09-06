import logging
import random
import unittest

import gymnasium as gym

logger = logging.getLogger()
logger.disabled = True

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


class RoomEnv2Test(unittest.TestCase):
    def test_all(self) -> None:
        env = gym.make("room_env:RoomEnv-v2", **config)
        (obs, question), info = env.reset()
        print(obs, question)
        while True:
            action_qa = question[0]
            action_explore = random.choice(["north", "east", "south", "west", "stay"])
            (obs, question), reward, done, truncated, info = env.step(
                (action_qa, action_explore)
            )
            if done:
                break
