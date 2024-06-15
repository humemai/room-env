import random
import unittest

import gymnasium as gym

from room_env.envs.room2 import *


class RandomizeObservationsTest(unittest.TestCase):
    def test_not_randomize_observations(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "none",
            "room_size": "xxl",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": True,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        obs_objects = []

        self.assertIn("room", observations["room"][0][0])
        self.assertEqual(observations["room"][0][1], "north")

        self.assertIn("room", observations["room"][1][0])
        self.assertEqual(observations["room"][1][1], "east")

        self.assertIn("room", observations["room"][2][0])
        self.assertEqual(observations["room"][2][1], "south")

        self.assertIn("room", observations["room"][3][0])
        self.assertEqual(observations["room"][3][1], "west")

        self.assertEqual(observations["room"][4][0], "agent")

        for obs in observations["room"][5:]:
            obs_objects.append(obs)

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["north", "east", "south", "west", "stay"]))
            )

            self.assertIn("room", observations["room"][0][0])
            self.assertEqual(observations["room"][0][1], "north")

            self.assertIn("room", observations["room"][1][0])
            self.assertEqual(observations["room"][1][1], "east")

            self.assertIn("room", observations["room"][2][0])
            self.assertEqual(observations["room"][2][1], "south")

            self.assertIn("room", observations["room"][3][0])
            self.assertEqual(observations["room"][3][1], "west")

            self.assertEqual(observations["room"][4][0], "agent")

            for obs in observations["room"][5:]:
                obs_objects.append(obs)

            if done:
                break

        for obs in obs_objects:
            self.assertEqual(obs[1], "atlocation")
            self.assertIn(obs[0].split("_")[0], ["sta", "ind", "dep"])

    def test_randomize_observations_all(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "all",
            "room_size": "xxl",
            "make_everything_static": True,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": True,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        heads_all_random = []
        observations, info = env.reset()
        for obs in observations["room"]:
            heads_all_random.append(obs[0])

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["stay"]))
            )
            for obs in observations["room"]:
                heads_all_random.append(obs[0])

            if done:
                break

        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "none",
            "room_size": "xxl",
            "make_everything_static": True,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": True,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        heads_all = []
        observations, info = env.reset()
        for obs in observations["room"]:
            heads_all.append(obs[0])

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["stay"]))
            )
            for obs in observations["room"]:
                heads_all.append(obs[0])

            if done:
                break

        self.assertNotEqual(heads_all, heads_all_random)
        self.assertEqual(len(heads_all), len(heads_all_random))
        self.assertEqual(sorted(heads_all), sorted(heads_all_random))

    def test_randomize_observations_objects(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "objects",
            "room_size": "xxl",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": True,
        }

        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        obs_objects = []

        self.assertIn("room", observations["room"][0][0])
        self.assertEqual(observations["room"][0][1], "north")

        self.assertIn("room", observations["room"][1][0])
        self.assertEqual(observations["room"][1][1], "east")

        self.assertIn("room", observations["room"][2][0])
        self.assertEqual(observations["room"][2][1], "south")

        self.assertIn("room", observations["room"][3][0])
        self.assertEqual(observations["room"][3][1], "west")

        self.assertEqual(observations["room"][4][0], "agent")

        obs_objects.append([obs[0].split("_")[0] for obs in observations["room"][5:]])
        for obs in observations["room"][5:]:
            self.assertEqual(obs[1], "atlocation")
            self.assertEqual(obs[2].split("_")[0], "room")
            self.assertIn(obs[0].split("_")[0], ["sta", "ind", "dep"])

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["north", "east", "south", "west", "stay"]))
            )

            self.assertIn("room", observations["room"][0][0])
            self.assertEqual(observations["room"][0][1], "north")

            self.assertIn("room", observations["room"][1][0])
            self.assertEqual(observations["room"][1][1], "east")

            self.assertIn("room", observations["room"][2][0])
            self.assertEqual(observations["room"][2][1], "south")

            self.assertIn("room", observations["room"][3][0])
            self.assertEqual(observations["room"][3][1], "west")

            self.assertEqual(observations["room"][4][0], "agent")

            obs_objects.append(
                [obs[0].split("_")[0] for obs in observations["room"][5:]]
            )
            for obs in observations["room"][5:]:
                self.assertEqual(obs[1], "atlocation")
                self.assertEqual(obs[2].split("_")[0], "room")
                self.assertIn(obs[0].split("_")[0], ["sta", "ind", "dep"])

            if done:
                break

    def test_randomize_observations_objects_no_wall(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "objects",
            "room_size": "xxl",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": False,
        }

        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        obs_objects = []
        obs_objects.append(observations["room"])

        for obs in observations["room"]:
            self.assertNotEqual(obs[2], "wall")

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["north", "east", "south", "west", "stay"]))
            )

            obs_objects.append(observations["room"])

            for obs in observations["room"]:
                self.assertNotEqual(obs[2], "wall")

            if done:
                break

    def test_randomize_observations_objects_middle(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "objects_middle",
            "room_size": "xxl",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": True,
        }

        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        obs_objects = []

        self.assertIn("room", observations["room"][0][0])
        self.assertEqual(observations["room"][0][1], "north")

        self.assertIn("room", observations["room"][1][0])
        self.assertEqual(observations["room"][1][1], "east")

        self.assertIn("room", observations["room"][2][0])
        self.assertEqual(observations["room"][2][1], "south")

        self.assertIn("room", observations["room"][3][0])
        self.assertEqual(observations["room"][3][1], "west")

        self.assertEqual(observations["room"][-1][0], "agent")

        obs_objects.append([obs[0].split("_")[0] for obs in observations["room"][4:-1]])
        for obs in observations["room"][4:-1]:
            self.assertEqual(obs[1], "atlocation")
            self.assertEqual(obs[2].split("_")[0], "room")
            self.assertIn(obs[0].split("_")[0], ["sta", "ind", "dep"])

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["north", "east", "south", "west", "stay"]))
            )

            self.assertIn("room", observations["room"][0][0])
            self.assertEqual(observations["room"][0][1], "north")

            self.assertIn("room", observations["room"][1][0])
            self.assertEqual(observations["room"][1][1], "east")

            self.assertIn("room", observations["room"][2][0])
            self.assertEqual(observations["room"][2][1], "south")

            self.assertIn("room", observations["room"][3][0])
            self.assertEqual(observations["room"][3][1], "west")

            self.assertEqual(observations["room"][-1][0], "agent")

            obs_objects.append(
                [obs[0].split("_")[0] for obs in observations["room"][4:-1]]
            )
            for obs in observations["room"][4:-1]:
                self.assertEqual(obs[1], "atlocation")
                self.assertEqual(obs[2].split("_")[0], "room")
                self.assertIn(obs[0].split("_")[0], ["sta", "ind", "dep"])

            if done:
                break

    def test_randomize_observations_objects_middle_no_wall(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "objects_middle",
            "room_size": "xxl",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": False,
        }

        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        obs_objects = []
        obs_objects.append(observations["room"])

        for obs in observations["room"]:
            self.assertNotEqual(obs[2], "wall")

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["north", "east", "south", "west", "stay"]))
            )

            obs_objects.append(observations["room"])

            for obs in observations["room"]:
                self.assertNotEqual(obs[2], "wall")

            if done:
                break
