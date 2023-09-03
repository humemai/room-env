"""Room environment compatible with gym."""
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
        """Entity, e.g., human, object, room.

        Args
        ----
        name: e.g., Tae, laptop, bed
        type: static, independent, or dependent
        init_probs: initial probabilities of being in a room
        transition_probs: transition probabilities of moving to another room

        """
        self.name = name
        self.type = type
        self.init_probs = init_probs
        self.transition_probs = transition_probs

        self.location = random.choices(
            list(self.init_probs.keys()),
            weights=list(self.init_probs.values()),
            k=1,
        )[0]

    def __repr__(self) -> str:
        return f"{self.type.title()}Object(name: {self.name}, location: {self.location}"

    def move_with_action(self, action: str, rooms: dict, current_location: str) -> str:
        """Move with action."""
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
        else:
            return current_location


class StaticObject(Object):
    def __init__(self, name: str, init_probs: dict, transition_probs: dict) -> None:
        super().__init__(name, "static", init_probs, transition_probs)

    def __repr__(self) -> str:
        return super().__repr__() + ")"


class IndepdentObject(Object):
    def __init__(
        self, name: str, init_probs: dict, transition_probs: dict, rooms: dict
    ) -> None:
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
        self.detach()

    def detach(self) -> None:
        """Detach from a dependent object."""
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
        super().__init__(name, "dependent", init_probs, transition_probs)
        self.independent_objects = independent_objects
        self.attach()

    def attach(self) -> None:
        """Attach to an independent object."""
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
        super().__init__(name, "agent", init_probs, transition_probs)
        self.rooms = rooms

    def move(self, action: str) -> None:
        """Agent can move north, east, south or west."""
        self.location = self.move_with_action(action, self.rooms, self.location)

    def __repr__(self) -> str:
        return "Agent(name: agent, location: " + self.location + ")"


class Room:
    def __init__(self, name: str, north: str, east: str, south: str, west: str) -> None:
        """Room.

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

    This environment is more formalized than the previous environments.
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
    ) -> None:
        """

        Args
        ----
        room_config: room configuration
        object_transition_config: object transition configuration
        object_init_config: object initial configuration
        question_prob: The probability of a question being asked at every observation.
        seed: random seed number

        """
        self.seed = seed
        seed_everything(self.seed)
        self.room_config = room_config
        self.object_transition_config = object_transition_config
        self.object_init_config = object_init_config
        self.question_prob = question_prob

        self._create_rooms()
        self._create_objects()

        # Our state / actionspace is quite complex. Here we just make a dummy spaces
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

    def _get_hidden_global_state(self) -> List[List[str]]:
        """Get global hidden state, i.e., list of triples, of the environment."""
        hidden_global_state = []
        for name, room in self.rooms.items():
            hidden_global_state.append([name, "tothenorth", room.north])
            hidden_global_state.append([name, "totheeast", room.east])
            hidden_global_state.append([name, "tothesouth", room.south])
            hidden_global_state.append([name, "tothewest", room.west])

        for obj_type in ["static", "independent", "dependent", "agent"]:
            for obj in self.objects[obj_type]:
                hidden_global_state.append([obj.name, "atlocation", obj.location])

        return hidden_global_state

    def get_observations(self) -> List[List[str]]:
        """Return what the agent sees in quadruples,

        i.e., (head, relation, tail, time).

        """
        agent_location = self.objects["agent"][0].location
        hidden_global_state = self._get_hidden_global_state()
        observations = []

        for triple in hidden_global_state:
            if triple[1] == "atlocation":
                if triple[2] == agent_location:
                    observations.append(triple)

            elif triple[1] in ["tothenorth", "totheeast", "tothesouth", "tothewest"]:
                if triple[0] == agent_location:
                    observations.append(triple)

            else:
                raise ValueError("Unknown relation.")

        for ob in observations:
            ob.append(self.current_time)

        return deepcopy(observations)

    def get_question(self) -> List[str]:
        """Uniformly sample a triple and ask a question.

        Returns
        -------
        question: [object, relation, tail], where one of object, relation, tail is
        replaced with ?

        """

        hidden_global_state = self._get_hidden_global_state()
        question = random.choice(hidden_global_state)

        idx = random.randint(0, len(question) - 1)
        self.question = question[:idx] + ["?"] + question[idx + 1 :]

        self.answers = []
        for triple in self._get_hidden_global_state():
            if self.question[0] == "?":
                if triple[1] == question[1] and triple[2] == question[2]:
                    self.answers.append(triple[0])
            elif self.question[1] == "?":
                if triple[0] == question[0] and triple[2] == question[2]:
                    self.answers.append(triple[1])
            elif self.question[2] == "?":
                if triple[0] == question[0] and triple[1] == question[1]:
                    self.answers.append(triple[2])
            else:
                raise ValueError("Unknown question.")

        if random.random() < self.question_prob:
            return deepcopy(self.question)
        else:
            return None

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

        return (self.get_observations(), self.get_question()), info

    def step(
        self, action_qa: str, action_explore: str
    ) -> Tuple[Tuple, int, bool, dict]:
        """An agent takes a set of actions.

        Args
        ----
        action_qa: An answer to the question.
        action_explore: An action to explore the environment, i.e., where to go.

        Returns
        -------
        (observation, question), reward, done, info

        """
        if action_qa in self.answers:
            reward = self.CORRECT
        else:
            reward = self.WRONG

        for obj in self.objects["independent"]:
            obj.move()

        for obj in self.objects["dependent"]:
            obj.attach()

        self.objects["agent"][0].move(action_explore)

        done = False
        info = {}

        self.current_time += 1

        return (self.get_observations(), self.get_question()), reward, done, info

    def render(self, mode="console") -> None:
        if mode != "console":
            raise NotImplementedError()
        else:
            pass
