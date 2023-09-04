"""RoomEnv2 environment compatible with gym.

This is the most complicated room environment so far. It has multiple rooms.
"""
import logging
import os
import random
from copy import deepcopy
from typing import List, Tuple, Dict

import gymnasium as gym

from ..utils import seed_everything

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Object:
    def __init__(
        self, name: str, type: str, init_probs: dict, transition_probs: dict
    ) -> None:
        """The simplest object class. One should inherit this class to make a more
        complex object.

        Args
        ----
        name: e.g., Alice, laptop, bed
        type: static, independent, dependent, or agent
        init_probs: initial probabilities of being in a room
        transition_probs: transition probabilities of moving to another room

        """
        self.name = name
        self.type = type
        self.init_probs = init_probs
        self.transition_probs = transition_probs

        # place an object in one of the rooms when it is created.
        self.location = random.choices(
            list(self.init_probs.keys()),
            weights=list(self.init_probs.values()),
            k=1,
        )[0]

    def __repr__(self) -> str:
        return f"{self.type.title()}Object(name: {self.name}, location: {self.location}"

    def move_with_action(self, action: str, rooms: dict, current_location: str) -> str:
        """Move with action.

        This method is only relevant for independent and agent objects, since they
        are the only ones that move with their will.

        Args
        ----
        action: north, east, south, west, or stay
        rooms: rooms
        current_location: current location

        Returns
        -------
        next_location: next location

        """
        assert action in ["north", "east", "south", "west", "stay"]
        if action == "north":
            next_location = rooms[current_location].north
        elif action == "east":
            next_location = rooms[current_location].east
        elif action == "south":
            next_location = rooms[current_location].south
        elif action == "west":
            next_location = rooms[current_location].west
        elif action == "stay":
            next_location = current_location

        if next_location != "wall":
            return next_location
        else:  # if the next location is a wall, stay.
            return current_location


class StaticObject(Object):
    def __init__(self, name: str, init_probs: dict, transition_probs: dict) -> None:
        """Static object does not move. One they are initialized, they stay forever.


        Args
        ----
        name: e.g., bed
        init_probs: initial probabilities of being in a room
        transition_probs: just a place holder. It's not gonna be used anyway.
        """
        super().__init__(name, "static", init_probs, transition_probs)

    def __repr__(self) -> str:
        return super().__repr__() + ")"


class IndepdentObject(Object):
    def __init__(
        self, name: str, init_probs: dict, transition_probs: dict, rooms: dict
    ) -> None:
        """Independent object moves to another room with the attached dependent objects.

        Args
        ----
        name: e.g., Alice
        init_probs: initial probabilities of being in a room
        transition_probs: transition probabilities of moving to another room
        rooms: rooms

        """
        super().__init__(name, "independent", init_probs, transition_probs)
        self.attached = []
        self.rooms = rooms

    def move(self) -> None:
        """Indendent object moves to another room with the attached dependent objects."""
        action = random.choices(
            list(self.transition_probs[self.location].keys()),
            weights=list(self.transition_probs[self.location].values()),
            k=1,
        )[0]

        self.location = self.move_with_action(action, self.rooms, self.location)

        for do in self.attached:
            do.location = self.location
        self.detach()  # detach the attached dependent objects after moving.

    def detach(self) -> None:
        """Detach from the dependent objects."""
        for do in self.attached:
            do.attached = None
        self.attached = []

    def __repr__(self) -> str:
        return super().__repr__() + f", attached: {[do.name for do in self.attached]})"


class DependentObject(Object):
    def __init__(
        self,
        name: str,
        init_probs: dict,
        transition_probs: dict,
        independent_objects: list,
    ) -> None:
        """Dependent object attaches to an independent object.

        It doesn't have the move method, since it moves with the independent object.

        Args
        ----
        name: e.g., laptop.
        init_probs: initial probabilities of being in a room.
        transition_probs: transition probabilities of moving to another room.
        independent_objects: independent objects in the environment.
        """
        super().__init__(name, "dependent", init_probs, transition_probs)
        self.independent_objects = independent_objects
        self.attach()  # attach to an independent object when it is created.

    def attach(self) -> None:
        """Attach to an independent object, with the provided randomness."""
        self.attached = None
        possible_attachments = []
        for io in self.independent_objects:
            if io.location == self.location:
                for io_name, prob in self.transition_probs.items():
                    if io.name == io_name:
                        if random.random() < prob:
                            possible_attachments.append(io)

        if len(possible_attachments) > 0:
            io = random.choice(possible_attachments)
            self.attached = io
            if self.name not in [do.name for do in io.attached]:
                io.attached.append(self)

    def __repr__(self) -> str:
        if self.attached is None:
            return super().__repr__() + ", attached: None)"
        else:
            return super().__repr__() + f", attached: {self.attached.name})"


class Agent(Object):
    def __init__(
        self, name: str, init_probs: dict, transition_probs: dict, rooms: dict
    ) -> None:
        """Agent class is the same as the independent object class, except that it
        moves with the provided action."""
        super().__init__(name, "agent", init_probs, transition_probs)
        self.rooms = rooms

    def move(self, action: str) -> None:
        """Agent can move north, east, south. west, or stay."""
        self.location = self.move_with_action(action, self.rooms, self.location)

    def __repr__(self) -> str:
        return "Agent(name: agent, location: " + self.location + ")"


class Room:
    def __init__(self, name: str, north: str, east: str, south: str, west: str) -> None:
        """Room. It has four sides and they can be either a wall or another room.

        Args
        ----
        name: e.g., officeroom, livingroom, bedroom
        north, east, south, west: either wall or another room

        """
        self.name = name
        self.north = north
        self.east = east
        self.south = south
        self.west = west

    def __repr__(self) -> str:
        return (
            f"Room(name: {self.name}, north: {self.north}, east: {self.east}, "
            f"south: {self.south}, west: {self.west})"
        )


class RoomEnv2(gym.Env):
    """the Room environment version 2.

    This environment is more formalized than the previous environments. Multiple rooms
    are supported. The agent can move north, east, south, west, or stay. Static,
    independent, dependent, agent objects are supported. Static objects do not move.
    Independent objects move with their will. Dependent objects move with independent
    objects. Agent moves with the provided action.

    Every string value is lower-cased to avoid confusion!!!

    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        room_config: dict,
        object_transition_config: dict,
        object_init_config: dict,
        question_prob: int = 1.0,
        seed: int = 42,
        terminates_at: int = 100,
    ) -> None:
        """

        Attributes
        ----------
        rooms: rooms: dict
        objects: objects: dict of lists
        question: question: list of strings
        answers: answers: list of strings
        current_time: current time: int

        Args
        ----
        room_config: room configuration
        object_transition_config: object transition configuration
        object_init_config: object initial configuration
        question_prob: The probability of a question being asked at every observation.
        seed: random seed number
        terminates_at: the environment terminates at this time step.

        """
        self.seed = seed
        seed_everything(self.seed)
        self.room_config = room_config
        self.object_transition_config = object_transition_config
        self.object_init_config = object_init_config
        self.question_prob = question_prob
        self.terminates_at = terminates_at

        self._create_rooms()
        self._create_objects()

        # Our state / actionspace are not tensors. Here we just make a dummy spaces
        # to bypass the gymnasium sanity check.
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

        self.CORRECT = 1
        self.WRONG = -1

    def _create_rooms(self) -> None:
        """Create rooms."""
        self.rooms = {}
        for name, config_ in self.room_config.items():
            self.rooms[name] = Room(name, **config_)

    def _create_objects(self) -> None:
        """Create objects."""
        self.objects = {"static": [], "independent": [], "dependent": [], "agent": []}

        for name, init_probs in self.object_init_config["static"].items():
            self.objects["static"].append(
                StaticObject(
                    name,
                    init_probs,
                    self.object_transition_config["static"][name],
                )
            )

        for name, init_probs in self.object_init_config["independent"].items():
            self.objects["independent"].append(
                IndepdentObject(
                    name,
                    init_probs,
                    self.object_transition_config["independent"][name],
                    self.rooms,
                )
            )

        for name, init_probs in self.object_init_config["dependent"].items():
            self.objects["dependent"].append(
                DependentObject(
                    name,
                    init_probs,
                    self.object_transition_config["dependent"][name],
                    self.objects["independent"],
                )
            )

        for name, init_probs in self.object_init_config["agent"].items():
            self.objects["agent"].append(
                Agent(
                    "agent",
                    init_probs,
                    self.object_transition_config["agent"][name],
                    self.rooms,
                )
            )

    def _get_hidden_global_state(self) -> None:
        """Get global hidden state, i.e., list of quadruples, of the environment.

        quadruples: [head, relation, tail, time]
        This is basically what the agent does not see and wants to estimate.

        """
        self.hidden_global_state = []
        for name, room in self.rooms.items():
            self.hidden_global_state.append([name, "tothenorth", room.north])
            self.hidden_global_state.append([name, "totheeast", room.east])
            self.hidden_global_state.append([name, "tothesouth", room.south])
            self.hidden_global_state.append([name, "tothewest", room.west])

        for obj_type in ["static", "independent", "dependent", "agent"]:
            for obj in self.objects[obj_type]:
                self.hidden_global_state.append([obj.name, "atlocation", obj.location])

        for triple in self.hidden_global_state:
            triple.append(self.current_time)

    def get_observations_and_question(self) -> Tuple[List[List[str]], List[str]]:
        """Return what the agent sees in quadruples, and the question.

        Returns
        -------
        observations: [head, relation, tail, time]
        question: [object, relation, tail, time], where one of object, relation, tail is
        replaced with ?

        """
        agent_location = self.objects["agent"][0].location
        self._get_hidden_global_state()
        self.observations = []

        for triple in self.hidden_global_state:  # atm, there are only 5 relations.
            if triple[1] == "atlocation":
                if triple[2] == agent_location:
                    self.observations.append(triple)

            elif triple[1] in ["tothenorth", "totheeast", "tothesouth", "tothewest"]:
                if triple[0] == agent_location:
                    self.observations.append(triple)

            else:
                raise ValueError("Unknown relation.")

        self.question = random.choice(self.hidden_global_state)

        idx = random.randint(0, len(self.question) - 2)
        self.question = self.question[:idx] + ["?"] + self.question[idx + 1 :]

        self.answers = []
        for triple in self.hidden_global_state:
            if self.question[0] == "?":
                if triple[1] == self.question[1] and triple[2] == self.question[2]:
                    self.answers.append(triple[0])
            elif self.question[1] == "?":
                if triple[0] == self.question[0] and triple[2] == self.question[2]:
                    self.answers.append(triple[1])
            elif self.question[2] == "?":
                if triple[0] == self.question[0] and triple[1] == self.question[1]:
                    self.answers.append(triple[2])
            else:
                raise ValueError(f"Unknown question: {self.question}")

        if random.random() >= self.question_prob:
            self.question = None

        return deepcopy(self.observations), deepcopy(self.question)

    def reset(self) -> Tuple[Tuple[list, list], dict]:
        """Reset the environment.


        Returns
        -------
        state, info

        """
        info = {}
        self._create_rooms()
        self._create_objects()
        self.current_time = 0

        return self.get_observations_and_question(), info

    def step(self, actions: Tuple[str, str]) -> Tuple[Tuple, int, bool, dict]:
        """An agent takes a set of actions.

        Args
        ----
        actions:
            action_qa: An answer to the question.
            action_explore: An action to explore the environment, i.e., where to go.
                north, east, south, west, or stay.

        Returns
        -------
        (observation, question), reward, done, info

        """
        action_qa, action_explore = actions
        if action_qa in self.answers:
            reward = self.CORRECT
        else:
            reward = self.WRONG

        for obj in self.objects["independent"]:
            obj.move()

        for obj in self.objects["dependent"]:
            obj.attach()

        self.objects["agent"][0].move(action_explore)

        if self.current_time < self.terminates_at:
            done = False
        else:
            done = True
        truncated = False
        info = {}

        self.current_time += 1

        return self.get_observations_and_question(), reward, done, truncated, info

    def render(self, mode="console") -> None:
        if mode != "console":
            raise NotImplementedError()
        else:
            pass
