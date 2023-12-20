import logging
import random
import unittest

import gymnasium as gym

from room_env.envs.room2 import *

logger = logging.getLogger()
logger.disabled = True


class QuestionProbTest(unittest.TestCase):
    def setUp(self) -> None:
        self.room_size = {
            "room_config": {
                "room_000": {
                    "north": "wall",
                    "east": "wall",
                    "south": "wall",
                    "west": "wall",
                }
            },
            "object_init_config": {
                "static": {"sta_000": {"room_000": 1}},
                "independent": {"ind_000": {"room_000": 1}},
                "dependent": {"dep_000": {"room_000": 1}},
                "agent": {"agent": {"room_000": 1}},
            },
            "object_transition_config": {
                "static": {"sta_000": None},
                "independent": {
                    "ind_000": {
                        "room_000": {
                            "north": 0,
                            "east": 0,
                            "south": 0,
                            "west": 0,
                            "stay": 1,
                        }
                    }
                },
                "dependent": {"dep_000": {"ind_000": 0.996476429155252}},
                "agent": {"agent": None},
            },
            "object_question_probs": {
                "static": {"sta_000": 1},
                "independent": {"ind_000": 0},
                "dependent": {"dep_000": 0},
                "agent": {"agent": 0.0},
            },
            "grid": [[1]],
            "room_indexes": [[0, 0]],
            "names": {
                "room": ["room_000"],
                "static_objects": ["sta_000"],
                "independent_objects": ["ind_000"],
                "dependent_objects": ["dep_000"],
            },
        }

    def test_always_one_object(self) -> None:
        self.room_size["object_question_probs"]["static"]["sta_000"] = 1.0
        self.room_size["object_question_probs"]["independent"]["ind_000"] = 0
        self.room_size["object_question_probs"]["dependent"]["dep_000"] = 0
        self.room_size["object_question_probs"]["agent"]["agent"] = 0

        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "all",
            "room_size": self.room_size,
            "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            "make_everything_static": False,
            "num_total_questions": 100,
            "question_interval": 1,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        while True:
            observations, reward, done, truncated, info = env.step(
                (
                    ["foo"],
                    random.choice(["north", "south", "east", "west", "stay"]),
                )
            )
            self.assertEqual(observations["questions"][0][0], "sta_000")

            if done:
                break

    def test_agent(self) -> None:
        self.room_size["object_question_probs"]["static"]["sta_000"] = 0.5
        self.room_size["object_question_probs"]["independent"]["ind_000"] = 0.49
        self.room_size["object_question_probs"]["dependent"]["dep_000"] = 0
        self.room_size["object_question_probs"]["agent"]["agent"] = 0.01

        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "all",
            "room_size": self.room_size,
            "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            "make_everything_static": False,
        }

        with self.assertRaises(AssertionError):
            env = gym.make("room_env:RoomEnv-v2", **env_config)
            observations, info = env.reset()

    def test_false_probs(self) -> None:
        self.room_size["object_question_probs"]["static"]["sta_000"] = 0.5
        self.room_size["object_question_probs"]["independent"]["ind_000"] = 0.49
        self.room_size["object_question_probs"]["dependent"]["dep_000"] = 0.02
        self.room_size["object_question_probs"]["agent"]["agent"] = 0

        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "all",
            "room_size": self.room_size,
            "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            "make_everything_static": False,
        }

        with self.assertRaises(AssertionError):
            env = gym.make("room_env:RoomEnv-v2", **env_config)
            observations, info = env.reset()

    def test_correct_probs(self) -> None:
        self.room_size["object_question_probs"]["static"]["sta_000"] = 0.333333
        self.room_size["object_question_probs"]["independent"]["ind_000"] = 0.333333
        self.room_size["object_question_probs"]["dependent"]["dep_000"] = 0.333333
        self.room_size["object_question_probs"]["agent"]["agent"] = 0.00

        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "objects",
            "room_size": self.room_size,
            "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            "make_everything_static": False,
        }

        questions = []

        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()
        questions.append(observations["questions"][0][0])

        while True:
            observations, reward, done, truncated, info = env.step(
                (
                    ["foo"],
                    random.choice(["north", "south", "east", "west", "stay"]),
                )
            )
            questions.append(observations["questions"][0][0])

            if done:
                break

        self.assertEqual(len(set(questions)), 3)
