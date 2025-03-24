import random
import unittest

import gymnasium as gym

from room_env.envs.room2 import *


class RoomEnv2OneRoomTest(unittest.TestCase):
    def test_all(self) -> None:
        for _ in range(100):
            room_size = {
                "room_config": {
                    "officeroom": {
                        "north": "wall",
                        "east": "wall",
                        "south": "wall",
                        "west": "wall",
                    }
                },
                "object_transition_config": {
                    "static": {"desk": None},
                    "independent": {},
                    "dependent": {},
                    "agent": {"agent": None},
                },
                "object_init_config": {
                    "static": {
                        "desk": {"officeroom": 1},
                    },
                    "independent": {},
                    "dependent": {},
                    "agent": {"agent": {"officeroom": 1.0}},
                },
                "object_question_probs": {
                    "static": {"desk": 1.0},
                    "agent": {"agent": 0},
                },
            }
            config = {
                "question_prob": 1.0,
                "seed": random.randint(0, 100000),
                "terminates_at": 9,
                "randomize_observations": "none",
                "room_size": room_size,
                "make_everything_static": False,
                "rewards": {"correct": 1, "wrong": -1, "partial": 0},
                "num_total_questions": 10,
                "question_interval": 1,
            }
            self.env = gym.make("room_env:RoomEnv-v2", **config)
            rewards = []
            observations, info = self.env.reset()
            self.assertEqual(info, {})
            self.assertEqual(
                self.env.rooms,
                {
                    "officeroom": Room(
                        name="officeroom",
                        north="wall",
                        east="wall",
                        south="wall",
                        west="wall",
                    )
                },
            )
            self.assertIn(["agent", "at_location", "officeroom"], observations["room"])
            self.assertIn(["desk", "at_location", "officeroom"], observations["room"])
            self.assertIn(["officeroom", "north", "wall"], observations["room"])
            self.assertIn(["officeroom", "east", "wall"], observations["room"])
            self.assertIn(["officeroom", "south", "wall"], observations["room"])
            self.assertIn(["officeroom", "west", "wall"], observations["room"])
            self.assertIn(
                observations["questions"][0],
                [
                    ["?", "at_location", "officeroom"],
                    ["desk", "at_location", "?"],
                ],
            )
            if observations["questions"][0] == ["?", "at_location", "officeroom"]:
                actions_qa = ["desk"]
            elif observations["questions"][0] == ["desk", "at_location", "?"]:
                actions_qa = ["officeroom"]
            else:
                raise ValueError(f"{observations['questions']}")

            question_previous = observations["questions"][0]

            observations, rewards_, done, truncated, info = self.env.step(
                (actions_qa, "east")
            )
            rewards.append(rewards_)
            self.assertEqual(rewards_, [1])
            self.assertFalse(done)
            # if question_previous == ["?", "at_location", "officeroom"]:
            #     self.assertEqual(info, {"answers": ["desk"], "timestamp": 0})
            if question_previous == ["desk", "at_location", "?"]:
                self.assertEqual(
                    info,
                    {
                        "answers": [{"current": "officeroom", "previous": None}],
                        "timestamp": 0,
                    },
                )
            else:
                raise ValueError

            self.assertIn(["agent", "at_location", "officeroom"], observations["room"])
            self.assertIn(["desk", "at_location", "officeroom"], observations["room"])
            self.assertIn(["officeroom", "north", "wall"], observations["room"])
            self.assertIn(["officeroom", "east", "wall"], observations["room"])
            self.assertIn(["officeroom", "south", "wall"], observations["room"])
            self.assertIn(["officeroom", "west", "wall"], observations["room"])
            self.assertIn(
                observations["questions"][0],
                [
                    ["?", "at_location", "officeroom"],
                    ["desk", "at_location", "?"],
                ],
            )

            if observations["questions"][0] == ["?", "at_location", "officeroom"]:
                actions_qa = ["desk"]
            elif observations["questions"][0] == ["desk", "at_location", "?"]:
                actions_qa = ["officeroom"]
            else:
                raise ValueError
            question_previous = observations["questions"][0]

            observations, rewards_, done, truncated, info = self.env.step(
                (actions_qa, "stay")
            )
            rewards.append(rewards_)
            self.assertEqual(rewards_, [1])
            self.assertFalse(done)
            # if question_previous == ["?", "at_location", "officeroom"]:
            #     self.assertEqual(info, {"answers": ["desk"], "timestamp": 1})
            if question_previous == ["desk", "at_location", "?"]:
                self.assertEqual(
                    info,
                    {
                        "answers": [{"current": "officeroom", "previous": None}],
                        "timestamp": 1,
                    },
                )
            else:
                raise ValueError

            for _ in range(7):
                observations, rewards_, done, truncated, info = self.env.step(
                    (["foo"], "stay")
                )
                rewards.append(rewards_)

                self.assertEqual(rewards_, [-1])
                self.assertFalse(done)

            observations, rewards_, done, truncated, info = self.env.step(
                (["bar"], "stay")
            )
            rewards.append(rewards_)
            self.assertEqual(len(rewards), 10)
            self.assertEqual(rewards_, [-1])
            self.assertTrue(done)
            # self.assertIsNone(self.env.observations_room)
            # self.assertIsNone(self.env.question)
            # self.assertIsNone(self.env.answers)


class RoomEnv2TwoRoomsTest(unittest.TestCase):
    def test_all(self) -> None:
        for _ in range(100):
            rewards = []
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
                    "agent": {"agent": None},
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
                    "agent": {"agent": {"officeroom": 1.0, "livingroom": 0}},
                },
                "object_question_probs": {
                    "static": {"desk": 0.5},
                    "independent": {"tae": 0.2},
                    "dependent": {"laptop": 0.3},
                    "agent": {"agent": 0},
                },
            }
            config = {
                "question_prob": 1.0,
                "seed": random.randint(0, 100000),
                "terminates_at": 99,
                "randomize_observations": "none",
                "room_size": room_size,
                "num_total_questions": 100,
                "rewards": {"correct": 1, "wrong": -1, "partial": 0},
            }
            self.env = gym.make("room_env:RoomEnv-v2", **config)

            observations, info = self.env.reset()
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
                observations["room"],
                [
                    ["officeroom", "north", "wall"],
                    ["officeroom", "east", "livingroom"],
                    ["officeroom", "south", "wall"],
                    ["officeroom", "west", "wall"],
                    ["agent", "at_location", "officeroom"],
                    ["desk", "at_location", "officeroom"],
                    ["tae", "at_location", "officeroom"],
                    ["laptop", "at_location", "officeroom"],
                ],
            )

            self.assertIn(
                observations["questions"][0],
                [
                    ["desk", "at_location", "?"],
                    ["tae", "at_location", "?"],
                    ["laptop", "at_location", "?"],
                    ["?", "at_location", "officeroom"],
                ],
            )

            if observations["questions"][0][0] == "?":
                actions_qa = [random.choice(["desk", "tae", "laptop"])]
            else:
                actions_qa = ["officeroom"]
            question_previous = observations["questions"]

            observations, rewards_, done, truncated, info = self.env.step(
                (actions_qa, "east")
            )
            rewards.append(rewards_)
            # if question_previous[0] == "?":
            #     self.assertEqual(
            #         info, {"answers": ["desk", "tae", "laptop"], "timestamp": 0}
            #     )
            if question_previous == ["desk", "at_location", "?"]:
                self.assertEqual(
                    info,
                    {
                        "answers": [{"current": "officeroom", "previous": None}],
                        "timestamp": 0,
                    },
                )

            self.assertEqual(
                observations["room"],
                [
                    ["livingroom", "north", "wall"],
                    ["livingroom", "east", "wall"],
                    ["livingroom", "south", "wall"],
                    ["livingroom", "west", "officeroom"],
                    ["agent", "at_location", "livingroom"],
                    ["tae", "at_location", "livingroom"],
                    ["laptop", "at_location", "livingroom"],
                ],
            )
            self.assertIn(
                observations["questions"][0],
                [
                    ["desk", "at_location", "?"],
                    ["tae", "at_location", "?"],
                    ["laptop", "at_location", "?"],
                    ["?", "at_location", "officeroom"],
                    ["?", "at_location", "livingroom"],
                ],
            )
            self.assertEqual(rewards_, [1])
            self.assertFalse(done)

            if observations["questions"][0] == ["desk", "at_location", "?"]:
                actions_qa = ["officeroom"]
            elif observations["questions"][0] == ["tae", "at_location", "?"]:
                actions_qa = ["livingroom"]
            elif observations["questions"][0] == ["laptop", "at_location", "?"]:
                actions_qa = ["livingroom"]
            elif observations["questions"][0] == ["?", "at_location", "officeroom"]:
                actions_qa = ["desk"]
            elif observations["questions"][0] == ["?", "at_location", "livingroom"]:
                actions_qa = [random.choice(["tae", "laptop"])]
            else:
                raise ValueError

            question_previous = observations["questions"][0]

            observations, rewards_, done, truncated, info = self.env.step(
                (actions_qa, "west")
            )
            rewards.append(rewards_)
            if question_previous == ["desk", "at_location", "?"]:
                self.assertEqual(
                    info,
                    {
                        "answers": [{"current": "officeroom", "previous": None}],
                        "timestamp": 1,
                    },
                )
            elif question_previous == [
                "tae",
                "at_location",
                "?",
            ] or question_previous == [
                "laptop",
                "at_location",
                "?",
            ]:
                self.assertEqual(
                    info,
                    {
                        "answers": [
                            {"current": "livingroom", "previous": "officeroom"}
                        ],
                        "timestamp": 1,
                    },
                )

            # elif question_previous == ["?", "at_location", "officeroom"]:
            #     self.assertEqual(info, {"answers": ["desk"], "timestamp": 1})
            # elif question_previous == ["?", "at_location", "livingroom"]:
            #     self.assertEqual(info, {"answers": ["tae", "laptop"], "timestamp": 1})
            else:
                raise ValueError

            self.assertEqual(
                observations["room"],
                [
                    ["officeroom", "north", "wall"],
                    ["officeroom", "east", "livingroom"],
                    ["officeroom", "south", "wall"],
                    ["officeroom", "west", "wall"],
                    ["agent", "at_location", "officeroom"],
                    ["desk", "at_location", "officeroom"],
                    ["tae", "at_location", "officeroom"],
                    ["laptop", "at_location", "officeroom"],
                ],
            )

            self.assertIn(
                observations["questions"][0],
                [
                    ["desk", "at_location", "?"],
                    ["tae", "at_location", "?"],
                    ["laptop", "at_location", "?"],
                    ["?", "at_location", "officeroom"],
                ],
            )
            self.assertEqual(rewards_, [1])
            self.assertFalse(done)

            for _ in range(97):
                observations, rewards_, done, truncated, info = self.env.step(
                    (["foo"], "stay")
                )
                rewards.append(rewards_)

                self.assertEqual(rewards_, [-1])
                self.assertFalse(done)

            observations, rewards_, done, truncated, info = self.env.step(
                (["bar"], "stay")
            )
            rewards.append(rewards_)
            self.assertEqual(len(rewards), 100)
            self.assertEqual(rewards_, [-1])
            self.assertTrue(done)
            # self.assertIsNone(self.env.observations_room)
            # self.assertIsNone(self.env.question)
            # self.assertIsNone(self.env.answers)


class RoomEnv2xxlTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = gym.make(
            "room_env:RoomEnv-v2",
            room_size="l",
            randomize_observations="none",
            include_walls_in_observations=True,
            num_total_questions=100,
        )

    def test_all(self) -> None:
        observations, info = self.env.reset()
        self.assertEqual(observations["room"][4][0], "agent")
        for obs in observations["room"][5:]:
            self.assertNotEqual(obs[0], "agent")
        while True:
            actions_qa = [random.choice(observations["questions"][0])]
            action_explore = random.choice(["north", "east", "south", "west", "stay"])
            observations, rewards_, done, truncated, info = self.env.step(
                (actions_qa, action_explore)
            )
            if done:
                break

            self.assertEqual(observations["room"][4][0], "agent")
            for obs in observations["room"][5:]:
                self.assertNotEqual(obs[0], "agent")


class RoomEnv2LayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = gym.make(
            "room_env:RoomEnv-v2",
            room_size="l",
            randomize_observations="none",
            include_walls_in_observations=True,
            num_total_questions=100,
        )

    def test_room_layout(self) -> None:
        observations, info = self.env.reset()

        room_layout = self.env.unwrapped.return_room_layout(exclude_walls=True)

        heads = [triple[0] for triple in room_layout]
        sorted_heads = sorted(heads)
        self.assertEqual(sorted_heads, heads)

        tails = [triple[2] for triple in room_layout]
        num_walls = tails.count("wall")
        self.assertEqual(num_walls, 0)

        room_layout = self.env.unwrapped.return_room_layout(exclude_walls=False)
        heads = [triple[0] for triple in room_layout]
        sorted_heads = sorted(heads)
        self.assertEqual(sorted_heads, heads)

        tails_ = [triple[2] for triple in room_layout]
        num_walls_ = tails_.count("wall")
        self.assertNotEqual(num_walls_, 0)

        self.assertGreater(num_walls_, num_walls)
