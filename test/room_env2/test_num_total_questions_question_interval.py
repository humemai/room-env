import unittest

import gymnasium as gym

from room_env.envs.room2 import *


class NumTotalQuestionsTest(unittest.TestCase):
    def test_false_config(self) -> None:
        env_config = {
            "question_prob": 0.5,
            "seed": 0,
            "terminates_at": 19,
            "randomize_observations": "objects",
            "room_size": "m",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            "num_total_questions": 99,
            "question_interval": 1,
        }
        with self.assertRaises(AssertionError):
            env = gym.make("room_env:RoomEnv-v2", **env_config)

        env_config["question_interval"] = 3
        with self.assertRaises(AssertionError):
            env = gym.make("room_env:RoomEnv-v2", **env_config)

    def test_wrong_answers(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 19,
            "randomize_observations": "all",
            "room_size": "m",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            "num_total_questions": 100,
            "question_interval": 2,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        self.assertEqual(observations["questions"], [])
        with self.assertRaises(AssertionError):
            observations, rewards, done, truncated, info = env.step(("foo", "stay"))
        with self.assertRaises(AssertionError):
            observations, rewards, done, truncated, info = env.step((["foo"], "stay"))

    def test_correct_answers(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 19,
            "randomize_observations": "all",
            "room_size": "m",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            "num_total_questions": 100,
            "question_interval": 2,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        questions_all = []
        self.assertEqual(len(observations["questions"]), 0)

        observations, rewards, done, truncated, info = env.step(([], "stay"))
        self.assertEqual(rewards, [])

        while True:
            flag = False
            if observations["questions"] == []:
                actions_qa = []
            else:
                flag = True
                actions_qa = ["foo"] * len(observations["questions"])
                for q in observations["questions"]:
                    questions_all.append(q)

            observations, rewards, done, truncated, info = env.step(
                (actions_qa, "stay")
            )
            if flag:
                self.assertEqual(rewards, [-1] * env.num_questions_step)
            else:
                self.assertEqual(rewards, [])

            if done:
                break

        self.assertEqual(len(questions_all), 100)
