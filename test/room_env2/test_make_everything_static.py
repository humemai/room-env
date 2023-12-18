import logging
import random
import unittest

import gymnasium as gym

from room_env.envs.room2 import *

logger = logging.getLogger()
logger.disabled = True


class MakeEverythingStaticTest(unittest.TestCase):
    def test_make_everything_static(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": False,
            "room_size": "l",
            "make_everything_static": True,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": True,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        initial_locations = {}
        for obj_type, objs in env.objects.items():
            for obj in objs:
                initial_locations[obj.name] = obj.location

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["stay"]))
            )

            for obj_type, objs in env.objects.items():
                for obj in objs:
                    self.assertEqual(obj.location, initial_locations[obj.name])
            if done:
                break

    def test_not_make_everything_static(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": False,
            "room_size": "l",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": -1, "partial": -1},
            "num_total_questions": 100,
            "question_interval": 1,
            "include_walls_in_observations": True,
        }
        locations_all = {}
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        for obj_type, objs in env.objects.items():
            for obj in objs:
                if obj.name in locations_all:
                    locations_all[obj.name].append(obj.location)
                else:
                    locations_all[obj.name] = [obj.location]

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["stay"]))
            )

            for obj_type, objs in env.objects.items():
                for obj in objs:
                    locations_all[obj.name].append(obj.location)
            if done:
                break

        all_lens = []
        for obj_name, locations in locations_all.items():
            all_lens.append(len(set(locations)))

        self.assertGreater(len(set(all_lens)), 1)
