import logging
import random
import unittest

import gymnasium as gym

from room_env.envs.room2 import *

logger = logging.getLogger()
logger.disabled = True


class RoomTest(unittest.TestCase):
    def test_all(self) -> None:
        with self.assertRaises(AssertionError):
            foo = Room(
                name="foo",
                north="bar",
                east="bar",
                south="bar",
                west="bar",
            )

        foo = Room(
            name="foo",
            north="room0",
            east="wall",
            south="wall",
            west="wall",
        )
        bar = Room(
            name="foo",
            north="room0",
            east="wall",
            south="wall",
            west="wall",
        )
        self.assertEqual(foo, bar)


class ObjectTest(unittest.TestCase):
    def test_all(self) -> None:
        with self.assertRaises(ValueError):
            foo = Object(
                name="foo",
                type="bar",
                init_probs={"room0": 1.0, "room1": 0.2},
                transition_probs={
                    "room0": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                    "room1": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                },
                question_prob=0.5,
            )

        foo = Object(
            name="foo",
            type="bar",
            init_probs={"room0": 1.0, "room1": 0},
            transition_probs={
                "room0": {
                    "north": 0.5,
                    "east": 0.5,
                    "south": 0,
                    "west": 0,
                    "stay": 0,
                },
                "room1": {
                    "north": 0,
                    "east": 0,
                    "south": 0.45,
                    "west": 0.55,
                    "stay": 0,
                },
            },
            question_prob=0.5,
        )

        bar = Object(
            name="foo",
            type="bar",
            init_probs={"room0": 1.0, "room1": 0},
            transition_probs={
                "room0": {
                    "north": 0.5,
                    "east": 0.5,
                    "south": 0,
                    "west": 0,
                    "stay": 0,
                },
                "room1": {
                    "north": 0,
                    "east": 0,
                    "south": 0.45,
                    "west": 0.55,
                    "stay": 0,
                },
            },
            question_prob=0.5,
        )
        self.assertEqual(foo, bar)

        rooms = {
            "room0": Room(
                name="room0", north="room1", east="wall", south="wall", west="wall"
            ),
            "room1": Room(
                name="room1", north="wall", east="wall", south="room0", west="wall"
            ),
        }
        with self.assertRaises(AssertionError):
            foo.move_with_action("foo", rooms, "foo")

        foo.location = "room0"
        foo.location = foo.move_with_action("north", rooms, "room0")
        self.assertEqual(foo.location, "room1")

        foo = Object(
            name="foo",
            type="bar",
            init_probs={"room0": 1.0, "room1": 0},
            transition_probs={
                "room0": {
                    "north": 0.5,
                    "east": 0.5,
                    "south": 0,
                    "west": 0,
                    "stay": 0,
                },
                "room1": {
                    "north": 0,
                    "east": 0,
                    "south": 0.45,
                    "west": 0.55,
                    "stay": 0,
                },
            },
            question_prob=0.5,
        )

        bar = Object(
            name="foo",
            type="bar",
            init_probs={"room0": 0, "room1": 1.0},
            transition_probs={
                "room0": {
                    "north": 0.5,
                    "east": 0.5,
                    "south": 0,
                    "west": 0,
                    "stay": 0,
                },
                "room1": {
                    "north": 0,
                    "east": 0,
                    "south": 0.45,
                    "west": 0.55,
                    "stay": 0,
                },
            },
            question_prob=0.5,
        )
        self.assertNotEqual(foo, bar)


class StaticObjectTest(unittest.TestCase):
    def test_all(self) -> None:
        with self.assertRaises(ValueError):
            foo = StaticObject(
                name="foo",
                init_probs={"room0": 1.0, "room1": 0.2},
                transition_probs={
                    "room0": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                    "room1": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                },
                question_prob=0.5,
            )
        with self.assertRaises(AssertionError):
            foo = StaticObject(
                name="foo",
                init_probs={"room0": 1.0, "room1": 0},
                transition_probs={
                    "room0": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                    "room1": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                },
                question_prob=0.5,
            )
        foo = StaticObject(
            name="foo",
            init_probs={"room0": 1.0, "room1": 0},
            transition_probs=None,
            question_prob=0.5,
        )
        bar = StaticObject(
            name="foo",
            init_probs={"room0": 1.0, "room1": 0},
            transition_probs=None,
            question_prob=0.5,
        )
        self.assertEqual(foo, bar)


class IndepdentObjectTest(unittest.TestCase):
    def test_all(self) -> None:
        with self.assertRaises(ValueError):
            foo = IndepdentObject(
                name="foo",
                init_probs={"room0": 1.0, "room1": 0.2},
                transition_probs={
                    "room0": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                    "room1": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                },
                rooms=None,
                question_prob=0.5,
            )

        with self.assertRaises(ValueError):
            foo = IndepdentObject(
                name="foo",
                init_probs={"room0": 0.35, "room1": 0.65},
                transition_probs={
                    "room0": {
                        "north": 0,
                        "east": 0.01,
                        "south": 0,
                        "west": 0,
                        "stay": 1.0,
                    },
                    "room1": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                },
                rooms=None,
                question_prob=0.5,
            )

        rooms = {
            "room0": Room(
                name="room0", north="room1", east="wall", south="wall", west="wall"
            ),
            "room1": Room(
                name="room1", north="wall", east="wall", south="room0", west="wall"
            ),
        }

        foo = IndepdentObject(
            name="foo",
            init_probs={"room0": 1, "room1": 0},
            transition_probs={
                "room0": {
                    "north": 1,
                    "east": 0,
                    "south": 0,
                    "west": 0,
                    "stay": 0,
                },
                "room1": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 1.0},
            },
            question_prob=0.5,
            rooms=rooms,
        )
        self.assertEqual(foo.location, "room0")
        foo.move()
        self.assertEqual(foo.location, "room1")
        for _ in range(100):
            foo.move()
            self.assertEqual(foo.location, "room1")


class DependentObjectTest(unittest.TestCase):
    def test_all(self) -> None:
        rooms = {
            "room0": Room(
                name="room0", north="wall", east="room1", south="wall", west="wall"
            ),
            "room1": Room(
                name="room1", north="wall", east="wall", south="room2", west="room0"
            ),
            "room2": Room(
                name="room2", north="room1", east="wall", south="wall", west="wall"
            ),
        }

        ind0 = IndepdentObject(
            name="ind0",
            init_probs={"room0": 1, "room1": 0, "room2": 0},
            transition_probs={
                "room0": {
                    "north": 0,
                    "east": 1,
                    "south": 0,
                    "west": 0,
                    "stay": 0,
                },
                "room1": {
                    "north": 0,
                    "east": 0,
                    "south": 1,
                    "west": 0,
                    "stay": 0,
                },
                "room2": {
                    "north": 0,
                    "east": 0,
                    "south": 0,
                    "west": 0,
                    "stay": 1.0,
                },
            },
            question_prob=0.5,
            rooms=rooms,
        )
        ind1 = IndepdentObject(
            name="ind1",
            init_probs={"room0": 1, "room1": 0, "room2": 0},
            transition_probs={
                "room0": {
                    "north": 0,
                    "east": 1,
                    "south": 0,
                    "west": 0,
                    "stay": 0,
                },
                "room1": {
                    "north": 0,
                    "east": 0,
                    "south": 1,
                    "west": 0,
                    "stay": 0,
                },
                "room2": {
                    "north": 0,
                    "east": 0,
                    "south": 0,
                    "west": 0,
                    "stay": 1.0,
                },
            },
            question_prob=0.5,
            rooms=rooms,
        )

        with self.assertRaises(ValueError):
            dep0 = DependentObject(
                name="dep0",
                init_probs={"room0": 1.0, "room1": 0.01, "room2": 0.0},
                transition_probs={"ind0": 1.0, "ind1": 0.0},
                independent_objects=[ind0, ind1],
                question_prob=0.5,
            )

        with self.assertRaises(ValueError):
            dep0 = DependentObject(
                name="dep0",
                init_probs={"room0": 1.0, "room1": 0.0, "room2": 0.0},
                transition_probs={"ind0": 1.1, "ind1": 0.0},
                independent_objects=[ind0, ind1],
                question_prob=0.5,
            )

        dep0 = DependentObject(
            name="dep0",
            init_probs={"room0": 1.0, "room1": 0.0, "room2": 0.0},
            transition_probs={"ind0": 1.0, "ind1": 0.0},
            independent_objects=[ind0, ind1],
            question_prob=0.5,
        )

        self.assertEqual(ind0.location, "room0")
        self.assertEqual(ind1.location, "room0")
        self.assertEqual(dep0.attached, ind0)
        self.assertEqual(dep0.location, "room0")

        for io in [ind0, ind1]:
            io.move()
        self.assertEqual(ind0.location, "room1")
        self.assertEqual(ind1.location, "room1")
        dep0.attach()
        self.assertEqual(dep0.attached, ind0)
        self.assertEqual(dep0.location, "room1")

        for io in [ind0, ind1]:
            io.move()
        self.assertEqual(ind0.location, "room2")
        self.assertEqual(ind1.location, "room2")
        dep0.attach()
        self.assertEqual(dep0.attached, ind0)
        self.assertEqual(dep0.location, "room2")

        for _ in range(10):
            for io in [ind0, ind1]:
                io.move()
            self.assertEqual(ind0.location, "room2")
            self.assertEqual(ind1.location, "room2")
            dep0.attach()
            self.assertEqual(dep0.attached, ind0)
            self.assertEqual(dep0.location, "room2")


class AgentTest(unittest.TestCase):
    def test_all(self) -> None:
        rooms = {
            "room0": Room(
                name="room0", north="wall", east="room1", south="wall", west="wall"
            ),
            "room1": Room(
                name="room1", north="wall", east="wall", south="room2", west="room0"
            ),
            "room2": Room(
                name="room2", north="room1", east="wall", south="wall", west="wall"
            ),
        }

        with self.assertRaises(ValueError):
            foo = Agent(
                name="foo",
                init_probs={"room0": 0.55, "room1": 0.455},
                transition_probs={
                    "room0": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                    "room1": {"north": 0, "east": 0, "south": 0, "west": 0, "stay": 0},
                },
                rooms=None,
                question_prob=0,
            )

        agent = Agent(
            name="foo",
            init_probs={"room0": 1.0, "room1": 0},
            transition_probs=None,
            rooms=rooms,
            question_prob=0,
        )
        agent_ = Agent(
            name="foo",
            init_probs={"room0": 1.0, "room1": 0},
            transition_probs=None,
            question_prob=0,
            rooms=rooms,
        )
        self.assertEqual(agent, agent_)

        self.assertEqual(agent.location, "room0")
        agent.move("north")
        self.assertEqual(agent.location, "room0")
        agent.move("south")
        self.assertEqual(agent.location, "room0")
        agent.move("west")
        self.assertEqual(agent.location, "room0")
        agent.move("stay")
        self.assertEqual(agent.location, "room0")
        agent.move("east")
        self.assertEqual(agent.location, "room1")
        agent.move("east")
        self.assertEqual(agent.location, "room1")
        agent.move("stay")
        self.assertEqual(agent.location, "room1")
        agent.move("west")
        self.assertEqual(agent.location, "room0")
        agent.move("east")
        self.assertEqual(agent.location, "room1")
        agent.move("south")
        self.assertEqual(agent.location, "room2")
        agent.move("east")
        self.assertEqual(agent.location, "room2")
        agent.move("south")
        self.assertEqual(agent.location, "room2")
        agent.move("west")
        self.assertEqual(agent.location, "room2")
        agent.move("north")
        self.assertEqual(agent.location, "room1")
        agent.move("south")
        self.assertEqual(agent.location, "room2")
        agent.move("stay")
        self.assertEqual(agent.location, "room2")
