import logging
import unittest

import gymnasium as gym

logger = logging.getLogger()
logger.disabled = True


class RoomEnv1Test(unittest.TestCase):
    def test_all(self) -> None:
        for des_size in ["xxs", "xs", "s", "m", "l"]:
            for question_prob in [0.1, 0.2, 0.4, 1]:
                for allow_random_human in [True, False]:
                    for allow_random_question in [True, False]:
                        for check_resources in [True, False]:
                            env = gym.make(
                                "room_env:RoomEnv-v1",
                                des_size=des_size,
                                question_prob=question_prob,
                                allow_random_human=allow_random_human,
                                allow_random_question=allow_random_question,
                                check_resources=check_resources,
                            )
                            (obs, question), info = env.reset()
                            while True:
                                (
                                    (obs, question),
                                    reward,
                                    done,
                                    truncated,
                                    info,
                                ) = env.step("foo")
                                if done:
                                    break
