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
        (observations, question), info = env.reset()
        self.assertEqual(observations[0][0], "agent")
        for obs in observations[1:]:
            self.assertNotEqual(obs[0], "agent")
        while True:
            action_qa = random.choice(question)
            action_explore = random.choice(["north", "east", "south", "west", "stay"])
            (observations, question), reward, done, truncated, info = env.step(
                (action_qa, action_explore)
            )
            self.assertEqual(observations[0][0], "agent")
            for obs in observations[1:]:
                self.assertNotEqual(obs[0], "agent")

            if done:
                break
