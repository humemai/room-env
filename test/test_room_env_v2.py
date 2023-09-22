import logging
import random
import unittest

import gymnasium as gym

logger = logging.getLogger()
logger.disabled = True

config = {"question_prob": 1.0, "seed": 42, "terminates_at": 99, "room_size": "dev"}


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
