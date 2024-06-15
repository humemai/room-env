import random
import unittest

import gymnasium as gym

from room_env.envs.room2 import *


class PartialPointsTest(unittest.TestCase):
    def test_partial_points(self) -> None:
        env_config = {
            "question_prob": 1.0,
            "seed": 0,
            "terminates_at": 99,
            "randomize_observations": "objects",
            "room_size": "m",
            "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            "make_everything_static": False,
            "num_total_questions": 100,
            "question_interval": 1,
        }
        env = gym.make("room_env:RoomEnv-v2", **env_config)
        observations, info = env.reset()

        while True:
            break_ = False
            for obj_type, objs in env.objects.items():
                for obj in objs:
                    if obj.name == observations["questions"][0][0]:
                        break_ = True
                    if break_:
                        break
                if break_:
                    break
            if len(set(obj.history)) == 1:
                observations, reward, done, truncated, info = env.step(
                    (
                        [obj.location],
                        random.choice(["north", "south", "east", "west", "stay"]),
                    )
                )

                self.assertEqual(reward, 1)
            else:
                for previous_location in obj.history[::-1]:
                    if previous_location != obj.location:
                        break

                observations, reward, done, truncated, info = env.step(
                    (
                        [previous_location],
                        random.choice(["north", "south", "east", "west", "stay"]),
                    )
                )
                self.assertEqual(reward, 0)

            if done:
                break
