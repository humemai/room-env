import random
import unittest

import gymnasium as gym

from room_env.envs.room2 import *


class IncludeWallsTest(unittest.TestCase):
    def test_include_walls(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "none",
            "room_size": "m",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": True,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()
        obs_wall_all = []
        obs_current_room = [
            obs
            for obs in observations["room"]
            if obs[1] in ["north", "east", "south", "west"]
        ]
        self.assertEqual(len(obs_current_room), 4)
        obs_wall = [obs for obs in observations["room"] if obs[2] == "wall"]

        for mem in obs_wall:
            self.assertIn(mem, obs_current_room)
            obs_wall_all.append(mem)

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["north", "east", "south", "west", "stay"]))
            )
            obs_current_room = [
                obs
                for obs in observations["room"]
                if obs[1] in ["north", "east", "south", "west"]
            ]
            self.assertEqual(len(obs_current_room), 4)
            obs_wall = [obs for obs in observations["room"] if obs[2] == "wall"]
            for mem in obs_wall:
                self.assertIn(mem, obs_current_room)
                obs_wall_all.append(mem)

            if done:
                break

        self.assertGreater(len(obs_wall_all), 0)

    def test_not_include_walls(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "none",
            "room_size": "m",
            "make_everything_static": True,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": False,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()
        obs_current_room = [
            obs
            for obs in observations["room"]
            if obs[1] in ["north", "east", "south", "west"]
        ]
        for mem in obs_current_room:
            self.assertNotEqual(mem[2], "wall")

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["north", "east", "south", "west", "stay"]))
            )
            obs_current_room = [
                obs
                for obs in observations["room"]
                if obs[1] in ["north", "east", "south", "west"]
            ]

            for mem in obs_current_room:
                self.assertNotEqual(mem[2], "wall")

            if done:
                break
