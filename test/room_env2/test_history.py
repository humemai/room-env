import random
import unittest
from copy import deepcopy

import gymnasium as gym

from room_env.envs.room2 import *


def all_elements_same(parent_list: list):
    # Check if the list is empty or contains only one element
    if not parent_list or len(parent_list) == 1:
        return True
    # Take the first element as a reference
    first_element = parent_list[0]
    # Compare every other element with the first one
    for element in parent_list[1:]:
        if element != first_element:
            return False
    return True


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

    def test_deterministic_objects(self) -> None:
        envs_all = {True: [], False: []}
        num_envs = 5
        for deterministic_objects in [False, True]:
            for seed in range(num_envs):
                env_config = {
                    "question_prob": 1.0,
                    "seed": seed,
                    "terminates_at": 99,
                    "randomize_observations": "objects",
                    "room_size": "l",
                    "make_everything_static": False,
                    "rewards": {"correct": 1, "wrong": 0, "partial": 0},
                    "num_total_questions": 100,
                    "question_interval": 1,
                    "include_walls_in_observations": True,
                    "deterministic_objects": deterministic_objects,
                }
                env = gym.make("room_env:RoomEnv-v2", **env_config)
                observations, info = env.reset()
                rewards = 0

                while True:
                    observations, reward, done, truncated, info = env.step(
                        (
                            ["random answer"] * len(observations["questions"]),
                            random.choice(["north", "east", "south", "west", "stay"]),
                        )
                    )
                    rewards += reward
                    if done or truncated:
                        break

                envs_all[deterministic_objects].append(deepcopy(env))

        num_static_objects = len(envs_all[True][0].objects["static"])
        num_independent_objects = len(envs_all[True][0].objects["independent"])
        num_dependent_objects = len(envs_all[True][0].objects["dependent"])

        for idx in range(num_envs):
            for obj_idx in range(num_static_objects):
                self.assertEqual(
                    envs_all[True][idx].objects["static"][obj_idx].history,
                    envs_all[False][idx].objects["static"][obj_idx].history,
                )

        for obj_idx in range(num_independent_objects):
            history = [
                envs_all[True][env_idx].objects["independent"][obj_idx].history
                for env_idx in range(num_envs)
            ]
            self.assertTrue(all_elements_same(history))

            history = [
                envs_all[False][env_idx].objects["independent"][obj_idx].history
                for env_idx in range(num_envs)
            ]
            self.assertFalse(all_elements_same(history))

        for obj_idx in range(num_dependent_objects):
            history = [
                envs_all[True][env_idx].objects["dependent"][obj_idx].history
                for env_idx in range(num_envs)
            ]
            self.assertTrue(all_elements_same(history))

            history = [
                envs_all[False][env_idx].objects["dependent"][obj_idx].history
                for env_idx in range(num_envs)
            ]
            self.assertFalse(all_elements_same(history))

        for env_idx in range(num_envs):
            for obj_idx in range(num_independent_objects):
                self.assertNotEqual(
                    envs_all[True][env_idx].objects["independent"][obj_idx].history,
                    envs_all[False][env_idx].objects["independent"][obj_idx].history,
                )
