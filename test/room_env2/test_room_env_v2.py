import logging
import random
import unittest

import gymnasium as gym

from room_env.envs.room2 import *

logger = logging.getLogger()
logger.disabled = True


class RoomEnv2MockTest(unittest.TestCase):
    def setUp(self) -> None:
        room_size = {
            "room_config": {
                "officeroom": {
                    "north": "wall",
                    "east": "livingroom",
                    "south": "wall",
                    "west": "wall",
                },
                "livingroom": {
                    "north": "wall",
                    "east": "wall",
                    "south": "wall",
                    "west": "officeroom",
                },
            },
            "object_transition_config": {
                "static": {"desk": None},
                "independent": {
                    "tae": {
                        "officeroom": {
                            "north": 0,
                            "east": 1.0,
                            "south": 0,
                            "west": 0,
                            "stay": 0,
                        },
                        "livingroom": {
                            "north": 0,
                            "east": 0,
                            "south": 0,
                            "west": 1.0,
                            "stay": 0,
                        },
                    },
                },
                "dependent": {
                    "laptop": {"tae": 1.0},
                },
                "agent": {"agent": {"officeroom": None, "livingroom": None}},
            },
            "object_init_config": {
                "static": {
                    "desk": {"officeroom": 1, "livingroom": 0},
                },
                "independent": {
                    "tae": {"officeroom": 1.0, "livingroom": 0},
                },
                "dependent": {
                    "laptop": {"officeroom": 1.0, "livingroom": 0},
                },
                "agent": {"agent": {"officeroom": 1.0, "livingroom": 0, "bedroom": 0}},
            },
        }
        config = {
            "question_prob": 1.0,
            "seed": random.randint(0, 100000),
            "terminates_at": 99,
            "randomize_observations": False,
            "room_size": room_size,
        }
        self.env = gym.make("room_env:RoomEnv-v2", **config)

    def test_all(self) -> None:
        for _ in range(100):
            (observations, question), info = self.env.reset()
            self.assertEqual(info, {})
            self.assertEqual(
                self.env.rooms,
                {
                    "officeroom": Room(
                        name="officeroom",
                        north="wall",
                        east="livingroom",
                        south="wall",
                        west="wall",
                    ),
                    "livingroom": Room(
                        name="livingroom",
                        north="wall",
                        east="wall",
                        south="wall",
                        west="officeroom",
                    ),
                },
            )
            self.assertEqual(
                observations,
                [
                    ["agent", "atlocation", "officeroom", 0],
                    ["desk", "atlocation", "officeroom", 0],
                    ["tae", "atlocation", "officeroom", 0],
                    ["laptop", "atlocation", "officeroom", 0],
                    ["officeroom", "north", "wall", 0],
                    ["officeroom", "east", "livingroom", 0],
                    ["officeroom", "south", "wall", 0],
                    ["officeroom", "west", "wall", 0],
                ],
            )
            self.assertIn(
                question,
                [
                    ["desk", "atlocation", "?", 0],
                    ["tae", "atlocation", "?", 0],
                    ["laptop", "atlocation", "?", 0],
                    ["?", "atlocation", "officeroom", 0],
                ],
            )

            if question[0] == "?":
                action_qa = random.choice(["desk", "tae", "laptop"])
            else:
                action_qa = "officeroom"
            question_previous = question

            (observations, question), reward, done, truncated, info = self.env.step(
                (action_qa, "east")
            )
            if question_previous[0] == "?":
                self.assertEqual(
                    info, {"answers": ["desk", "tae", "laptop"], "timestamp": 0}
                )
            else:
                self.assertEqual(info, {"answers": ["officeroom"], "timestamp": 0})

            self.assertEqual(
                observations,
                [
                    ["agent", "atlocation", "livingroom", 1],
                    ["tae", "atlocation", "livingroom", 1],
                    ["laptop", "atlocation", "livingroom", 1],
                    ["livingroom", "north", "wall", 1],
                    ["livingroom", "east", "wall", 1],
                    ["livingroom", "south", "wall", 1],
                    ["livingroom", "west", "officeroom", 1],
                ],
            )
            self.assertIn(
                question,
                [
                    ["desk", "atlocation", "?", 1],
                    ["tae", "atlocation", "?", 1],
                    ["laptop", "atlocation", "?", 1],
                    ["?", "atlocation", "officeroom", 1],
                    ["?", "atlocation", "livingroom", 1],
                ],
            )
            self.assertEqual(reward, 1)
            self.assertFalse(done)

            if question == ["desk", "atlocation", "?", 1]:
                action_qa = "officeroom"
            elif question == ["tae", "atlocation", "?", 1]:
                action_qa = "livingroom"
            elif question == ["laptop", "atlocation", "?", 1]:
                action_qa = "livingroom"
            elif question == ["?", "atlocation", "officeroom", 1]:
                action_qa = "desk"
            elif question == ["?", "atlocation", "livingroom", 1]:
                action_qa = random.choice(["tae", "laptop"])
            else:
                raise ValueError

            question_previous = question

            (observations, question), reward, done, truncated, info = self.env.step(
                (action_qa, "west")
            )
            if question_previous == ["desk", "atlocation", "?", 1]:
                self.assertEqual(info, {"answers": ["officeroom"], "timestamp": 1})
            elif question_previous == [
                "tae",
                "atlocation",
                "?",
                1,
            ] or question_previous == [
                "laptop",
                "atlocation",
                "?",
                1,
            ]:
                self.assertEqual(info, {"answers": ["livingroom"], "timestamp": 1})
            elif question_previous == ["?", "atlocation", "officeroom", 1]:
                self.assertEqual(info, {"answers": ["desk"], "timestamp": 1})
            elif question_previous == ["?", "atlocation", "livingroom", 1]:
                self.assertEqual(info, {"answers": ["tae", "laptop"], "timestamp": 1})
            else:
                raise ValueError

            self.assertEqual(
                observations,
                [
                    ["agent", "atlocation", "officeroom", 2],
                    ["desk", "atlocation", "officeroom", 2],
                    ["tae", "atlocation", "officeroom", 2],
                    ["laptop", "atlocation", "officeroom", 2],
                    ["officeroom", "north", "wall", 2],
                    ["officeroom", "east", "livingroom", 2],
                    ["officeroom", "south", "wall", 2],
                    ["officeroom", "west", "wall", 2],
                ],
            )

            self.assertIn(
                question,
                [
                    ["desk", "atlocation", "?", 2],
                    ["tae", "atlocation", "?", 2],
                    ["laptop", "atlocation", "?", 2],
                    ["?", "atlocation", "officeroom", 2],
                ],
            )
            self.assertEqual(reward, 1)
            self.assertFalse(done)


# class RoomEnv2DevTest(unittest.TestCase):
#     def setUp(self) -> None:
#         self.env = gym.make("room_env:RoomEnv-v2", room_size="dev")

#     def test_all(self) -> None:
#         (observations, question), info = self.env.reset()
#         self.assertEqual(observations[0][0], "agent")
#         for obs in observations[1:]:
#             self.assertNotEqual(obs[0], "agent")
#         while True:
#             action_qa = random.choice(question)
#             action_explore = random.choice(["north", "east", "south", "west", "stay"])
#             (observations, question), reward, done, truncated, info = self.env.step(
#                 (action_qa, action_explore)
#             )
#             self.assertEqual(observations[0][0], "agent")
#             for obs in observations[1:]:
#                 self.assertNotEqual(obs[0], "agent")

#             if done:
#                 break
