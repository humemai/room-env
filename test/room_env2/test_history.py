import logging
import random
import unittest

import gymnasium as gym

from room_env.envs.room2 import *

logger = logging.getLogger()
logger.disabled = True


class ObjectHistoryTest(unittest.TestCase):
    def test_all_static(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "all",
            "room_size": "m",
            "make_everything_static": True,
            "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            "num_total_questions": 100,
            "question_interval": 1,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        locations = {}
        for obj_type, objs in env.objects.items():
            for obj in objs:
                if obj.name in locations:
                    locations[obj.name].append(obj.location)
                else:
                    locations[obj.name] = [obj.location]

        while True:
            observations, reward, done, truncated, info = env.step((["foo"], "stay"))
            for obj_type, objs in env.objects.items():
                for obj in objs:
                    locations[obj.name].append(obj.location)
            if done:
                break

        for obj_type, objs in env.objects.items():
            for obj in objs:
                self.assertEqual(len(locations[obj.name]), 101)
                self.assertEqual(obj.history, locations[obj.name])
                self.assertEqual(len(set(locations[obj.name])), 1)

    def test_not_all_static(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "all",
            "room_size": "m",
            "make_everything_static": False,
            "num_total_questions": 100,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        locations = {}
        for obj_type, objs in env.objects.items():
            for obj in objs:
                if obj.name in locations:
                    locations[obj.name].append(obj.location)
                else:
                    locations[obj.name] = [obj.location]

        while True:
            observations, reward, done, truncated, info = env.step(
                (["foo"], random.choice(["north", "south", "east", "west"]))
            )
            for obj_type, objs in env.objects.items():
                for obj in objs:
                    locations[obj.name].append(obj.location)

            if done:
                break

        for obj_type, objs in env.objects.items():
            for obj in objs:
                self.assertEqual(len(locations[obj.name]), 101)
                self.assertEqual(obj.history, locations[obj.name])

                if obj_type == "static":
                    self.assertEqual(len(set(locations[obj.name])), 1)
                elif obj_type in ["independent", "agent"]:
                    self.assertNotEqual(len(set(locations[obj.name])), 1)
                else:
                    pass
