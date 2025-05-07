import os
import random
from copy import deepcopy
from pprint import pprint

import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import clear_output

# -----------------------------------------------------------------
# NEW OR MODIFIED
from rdflib import Graph, Literal, URIRef, Namespace, URIRef

# -----------------------------------------------------------------

from ..utils import is_running_notebook
from ..utils import read_json_prod as read_json
from ..utils import sample_max_value_key, seed_everything

EPSILON = 1e-3


# NEW: helper function for converting an rdflib.Graph to a Python dict
def rdf_to_dict(g: Graph) -> dict:
    """Convert an rdflib Graph to a Python dictionary representation."""
    graph_dict = {}
    for s, p, o in g:
        s_key = str(s)  # Convert subject to string
        p_key = str(p)  # Convert predicate to string
        o_value = str(o)  # Convert object to string

        # Initialize subject entry if not present
        if s_key not in graph_dict:
            graph_dict[s_key] = {}

        # If predicate exists, store as a list (for multiple values)
        if p_key in graph_dict[s_key]:
            if isinstance(graph_dict[s_key][p_key], list):
                graph_dict[s_key][p_key].append(o_value)
            else:
                graph_dict[s_key][p_key] = [graph_dict[s_key][p_key], o_value]
        else:
            graph_dict[s_key][p_key] = o_value

    return graph_dict


class Object:
    """
    Base class representing any object in the room environment.

    This class serves as a parent for all object types (static, independent, dependent, agent).
    It handles basic properties like name, location, and movement logic common to all objects.

    Attributes:
        name (str): The unique identifier for the object.
        type (str): The type of object (static, independent, dependent, agent).
        init_probs (dict): Initial probabilities for starting locations.
        transition_probs (dict): Transition probabilities for movement.
        question_prob (float): Probability of being chosen for a question.
        deterministic (bool): Whether object behavior is deterministic.
        location (str): Current location of the object.
        history (list): History of locations the object has been in.
    """

    def __init__(
        self,
        name: str,
        type: str,
        init_probs: dict,
        transition_probs: dict,
        question_prob: float,
        deterministic: bool,
    ) -> None:
        """
        Initialize an Object.

        Args:
            name: Unique identifier for this object.
            type: Type of object (static, independent, dependent, agent).
            init_probs: Dictionary mapping room names to initial placement
                probabilities.
            transition_probs: Dictionary of movement transition probabilities.
            question_prob: Probability of being chosen for a question.
            deterministic: Whether behavior should be deterministic.

        Raises:
            ValueError: If initial probabilities don't sum to 1.
            AssertionError: If question_prob is not between 0 and 1.
        """
        self.name = name
        self.type = type
        self.init_probs = init_probs
        self.history = []

        if abs(sum(self.init_probs.values()) - 1) >= EPSILON:
            raise ValueError(
                f"The sum of the initial probabilities must be 1. "
                f"but it's {sum(self.init_probs.values())}"
            )
        self.transition_probs = transition_probs
        self.question_prob = question_prob
        self.deterministic = deterministic

        assert (
            0 <= self.question_prob <= 1
        ), f"question_prob must be between 0 and 1, but it's {self.question_prob}"

        # place an object in one of the rooms when it is created.
        if self.deterministic:
            self.location = sample_max_value_key(self.init_probs)
        else:
            self.location = random.choices(
                list(self.init_probs.keys()),
                weights=list(self.init_probs.values()),
                k=1,
            )[0]

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Returns:
            String representation showing object type, name, and location.
        """
        return f"{self.type.title()}Object(name: {self.name}, location: {self.location}"

    def __eq__(self, other) -> bool:
        """
        Check equality between this object and another.

        Args:
            other: Another Object instance to compare with.

        Returns:
            True if objects have the same properties, False otherwise.
        """
        return (
            self.name == other.name
            and self.type == other.type
            and self.init_probs == other.init_probs
            and self.transition_probs == other.transition_probs
            and self.location == other.location
        )

    def move_with_action(self, action: str, rooms: dict, current_location: str) -> str:
        """
        Determine the next location based on a given action.

        Args:
            action: Direction to move ('north', 'east', 'south', 'west', 'stay').
            rooms: Dictionary of Room objects.
            current_location: Current location name.

        Returns:
            Name of the next location after taking the action.

        Raises:
            AssertionError: If action is not valid.
        """
        assert action in [
            "north",
            "east",
            "south",
            "west",
            "stay",
        ], f"{action} is not valid"
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

        return next_location if next_location != "wall" else current_location

    def _update_history(self) -> None:
        """
        Update the location history of this object.

        Appends the current location to the history list.
        """
        self.history.append(self.location)


class StaticObject(Object):
    """
    Represents a static object that stays in one place.

    Static objects do not move during the simulation.

    Attributes:
        Same as Object class, but transition_probs is always None.
    """

    def __init__(
        self,
        name: str,
        init_probs: dict,
        transition_probs: dict,
        question_prob: float,
        deterministic: bool,
    ) -> None:
        """
        Initialize a StaticObject.

        Args:
            name: Unique identifier for this object.
            init_probs: Dictionary mapping room names to initial placement probabilities.
            transition_probs: Must be None for static objects.
            question_prob: Probability of being chosen for a question.
            deterministic: Whether behavior should be deterministic.

        Raises:
            AssertionError: If transition_probs is not None.
        """
        super().__init__(
            name,
            "static",
            init_probs,
            transition_probs,
            question_prob,
            deterministic=deterministic,
        )
        # for a static object, we don't use transition_probs
        assert self.transition_probs is None, "Static objects do not move."

    def __repr__(self) -> str:
        """
        Return a string representation of the static object.

        Returns:
            String representation of the static object.
        """
        return super().__repr__() + ")"


class IndepdentObject(Object):
    """
    Represents an independent object that can move on its own.

    Independent objects move according to their transition probabilities
    and can have dependent objects attached to them.

    Attributes:
        rooms (dict): Dictionary of Room objects.
        attached (list): List of dependent objects attached to this object.
    """

    def __init__(
        self,
        name: str,
        init_probs: dict,
        transition_probs: dict,
        rooms: dict,
        question_prob: float,
        deterministic: bool,
    ) -> None:
        """
        Initialize an IndepdentObject.

        Args:
            name: Unique identifier for this object.
            init_probs: Dictionary mapping room names to initial placement probabilities.
            transition_probs: Dictionary of movement transition probabilities.
            rooms: Dictionary of Room objects.
            question_prob: Probability of being chosen for a question.
            deterministic: Whether behavior should be deterministic.

        Raises:
            ValueError: If transition probabilities for any state don't sum to 1.
        """
        super().__init__(
            name,
            "independent",
            init_probs,
            transition_probs,
            question_prob,
            deterministic,
        )
        for key, val in self.transition_probs.items():
            if abs(sum(val.values()) - 1) >= EPSILON:
                raise ValueError(
                    "The sum of the transition probabilities for an independent object "
                    f"must be 1. but it's {sum(val.values())}"
                )
        self.attached = []
        self.rooms = rooms

    def move(self) -> None:
        """
        Move the independent object according to its transition probabilities.

        Also moves any attached dependent objects to the same new location.
        """
        if self.deterministic:
            # pick the best action by max prob, ignoring "stay"
            action = sample_max_value_key(
                self.transition_probs[self.location], keys_to_exclude=["stay"]
            )
        else:
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
        """
        Detach all dependent objects from this independent object.

        Clears the attachment relationships in both directions.
        """
        for do in self.attached:
            do.attached = None
        self.attached = []

    def __repr__(self) -> str:
        """
        Return a string representation of the independent object.

        Returns:
            String representation including attached objects.
        """
        return super().__repr__() + f", attached: {[do.name for do in self.attached]})"

    def __eq__(self, other) -> bool:
        """
        Check equality between this object and another.

        Args:
            other: Another IndepdentObject instance to compare with.

        Returns:
            True if objects have the same properties, False otherwise.
        """
        return (
            super().__eq__(other)
            and self.attached == other.attached
            and self.rooms == other.rooms
        )


class DependentObject(Object):
    """
    Represents a dependent object that can attach to independent objects.

    Dependent objects move when attached to an independent object that moves.
    They can attach to independent objects in the same room based on probabilities.

    Attributes:
        independent_objects (list): List of independent objects in the environment.
        attached (IndepdentObject or None): The independent object this object is attached to.
    """

    def __init__(
        self,
        name: str,
        init_probs: dict,
        transition_probs: dict,
        independent_objects: list,
        question_prob: float,
        deterministic: bool,
    ) -> None:
        """
        Initialize a DependentObject.

        Args:
            name: Unique identifier for this object.
            init_probs: Dictionary mapping room names to initial placement probabilities.
            transition_probs: Dictionary mapping independent object names to attachment probabilities.
            independent_objects: List of independent objects in the environment.
            question_prob: Probability of being chosen for a question.
            deterministic: Whether behavior should be deterministic.

        Raises:
            ValueError: If any transition probability is greater than 1.
        """
        super().__init__(
            name,
            "dependent",
            init_probs,
            transition_probs,
            question_prob,
            deterministic,
        )
        for key, val in self.transition_probs.items():
            if val >= 1 + EPSILON:
                raise ValueError(
                    "The transition probability for a dependent object must "
                    f"be <= 1. but it's {val}"
                )
        self.independent_objects = independent_objects
        self.attach()

    def attach(self) -> None:
        """
        Try to attach to an independent object in the same room.

        The attachment is probabilistic based on transition_probs unless deterministic is True.
        """
        self.attached = None
        possible_attachments = []
        for io in self.independent_objects:
            if io.location == self.location:
                for io_name, prob in self.transition_probs.items():
                    if io.name == io_name:
                        if self.deterministic:
                            possible_attachments.append(io)
                        else:
                            if random.random() < prob:
                                possible_attachments.append(io)

        if len(possible_attachments) > 0:
            if self.deterministic:
                io = possible_attachments[0]
            else:
                io = random.choice(possible_attachments)
            self.attached = io
            if self.name not in [do.name for do in io.attached]:
                io.attached.append(self)

    def __repr__(self) -> str:
        """
        Return a string representation of the dependent object.

        Returns:
            String representation including attachment information.
        """
        if self.attached is None:
            return super().__repr__() + ", attached: None)"
        else:
            return super().__repr__() + f", attached: {self.attached.name})"

    def __eq__(self, other) -> bool:
        """
        Check equality between this object and another.

        Args:
            other: Another DependentObject instance to compare with.

        Returns:
            True if objects have the same properties, False otherwise.
        """
        return (
            super().__eq__(other)
            and self.attached == other.attached
            and self.independent_objects == other.independent_objects
        )


class Agent(Object):
    """
    Represents an agent that can be controlled through actions.

    The agent moves according to external actions rather than transition probabilities.

    Attributes:
        rooms (dict): Dictionary of Room objects.
    """

    def __init__(
        self,
        name: str,
        init_probs: dict,
        transition_probs: dict,
        rooms: dict,
        question_prob: float,
    ) -> None:
        """
        Initialize an Agent.

        Args:
            name: Unique identifier for this agent.
            init_probs: Dictionary mapping room names to initial placement probabilities.
            transition_probs: Must be None for agents.
            rooms: Dictionary of Room objects.
            question_prob: Must be close to 0 as agents are not questionable.

        Raises:
            AssertionError: If transition_probs is not None or question_prob is not close to 0.
        """
        assert abs(question_prob) <= EPSILON, "Agents are not questionable."

        super().__init__(
            name,
            "agent",
            init_probs,
            transition_probs,
            question_prob,
            deterministic=False,
        )
        assert self.transition_probs is None, "Agent objects do not move by itself."
        self.rooms = rooms

    def move(self, action: str) -> None:
        """
        Move the agent according to the specified action.

        Args:
            action: Direction to move ('north', 'east', 'south', 'west', 'stay').
        """
        self.location = self.move_with_action(action, self.rooms, self.location)

    def __repr__(self) -> str:
        """
        Return a string representation of the agent.

        Returns:
            String representation of the agent.
        """
        return "Agent(name: agent, location: " + self.location + ")"

    def __eq__(self, other) -> bool:
        """
        Check equality between this agent and another.

        Args:
            other: Another Agent instance to compare with.

        Returns:
            True if agents have the same properties, False otherwise.
        """
        return super().__eq__(other) and self.rooms == other.rooms


class Room:
    """
    Represents a room in the environment.

    Rooms are connected to each other in the four cardinal directions.

    Attributes:
        name (str): Unique name of the room.
        north (str): Name of the room to the north, or 'wall'.
        east (str): Name of the room to the east, or 'wall'.
        south (str): Name of the room to the south, or 'wall'.
        west (str): Name of the room to the west, or 'wall'.
    """

    def __init__(self, name: str, north: str, east: str, south: str, west: str) -> None:
        """
        Initialize a Room.

        Args:
            name: Unique name for this room.
            north: Name of the room to the north, or 'wall'.
            east: Name of the room to the east, or 'wall'.
            south: Name of the room to the south, or 'wall'.
            west: Name of the room to the west, or 'wall'.

        Raises:
            AssertionError: If the room layout is invalid (e.g., duplicate connections).
        """
        self.name = name
        self.north = north
        self.east = east
        self.south = south
        self.west = west

        rooms_walls = [self.north, self.east, self.south, self.west]
        rooms_walls = [rw for rw in rooms_walls if rw != "wall"]
        assert len(set(rooms_walls)) == len(rooms_walls), "room layout wrong."

    def __repr__(self) -> str:
        """
        Return a string representation of the room.

        Returns:
            String representation showing room name and connections.
        """
        return (
            f"Room(name: {self.name}, north: {self.north}, east: {self.east}, "
            f"south: {self.south}, west: {self.west})"
        )

    def __eq__(self, other) -> bool:
        """
        Check equality between this room and another.

        Args:
            other: Another Room instance to compare with.

        Returns:
            True if rooms have the same properties, False otherwise.
        """
        return (
            self.name == other.name
            and self.north == other.north
            and self.east == other.east
            and self.south == other.south
            and self.west == other.west
        )


class RoomEnv2(gym.Env):
    """
    The Room environment version 2.

    A gymnasium environment for simulating objects and an agent in interconnected rooms.
    Objects can move independently or be attached to other objects, and the agent can be
    controlled through actions. The environment supports question-answering about object locations.

    Attributes:
        rooms (dict): Dictionary of Room objects.
        objects (dict): Dictionary containing lists of objects by type.
        relations (list): List of relation types in the environment.
        entities (dict): Dictionary of entity names by type.
        hidden_global_state (list): Current state as triples.
        observations_room (list): Current observations of the room.
        questions (list): Current questions.
        answers (list): Current answers to questions.
    """

    def __init__(
        self,
        question_prob: int = 1.0,
        seed: int = 42,
        terminates_at: int = 99,
        randomize_observations: str = "all",
        room_size: str = "xl-different-prob",
        rewards: dict = {"correct": 1, "wrong": 0, "partial": 0},
        make_everything_static: bool = False,
        num_total_questions: int = 1000,
        question_interval: int = 1,
        include_walls_in_observations: bool = True,
        deterministic_objects: bool = False,
    ) -> None:
        """
        Initialize the Room environment.

        Args:
            question_prob: Probability of generating a question when appropriate.
            seed: Random seed for reproducibility.
            terminates_at: Number of steps before the episode terminates.
            randomize_observations: How to randomize observations ('all', 'objects', 'objects_middle', 'none').
            room_size: Size configuration of the room layout.
            rewards: Dictionary specifying rewards for correct, wrong, and partial answers.
            make_everything_static: If True, objects don't move.
            num_total_questions: Total number of questions to generate throughout all episodes.
            question_interval: Interval between generating questions.
            include_walls_in_observations: Whether to include walls in observations.
            deterministic_objects: Whether objects should behave deterministically.

        Raises:
            AssertionError: If the total questions or terminates_at don't align with question_interval.
        """
        super().__init__()
        self.is_notebook = is_running_notebook()
        if isinstance(room_size, str):
            config_all = read_json(f"./data/room-config-{room_size}-v2.json")
        else:
            for key in [
                "object_init_config",
                "object_transition_config",
                "room_config",
            ]:
                assert key in room_size, f"{key} is not in the room_size dict."
            config_all = room_size

        self.room_config = config_all["room_config"]
        if "mapping" in config_all:
            self.mapping = config_all["mapping"]
        self.object_transition_config = config_all["object_transition_config"]
        self.object_init_config = config_all["object_init_config"]
        self.object_question_probs = config_all["object_question_probs"]

        if "grid" in config_all:
            self.grid = config_all["grid"]
        if "room_indexes" in config_all:
            self.room_indexes = config_all["room_indexes"]
        if "names" in config_all:
            self.names = config_all["names"]

        self.seed = seed
        seed_everything(self.seed)
        self.question_prob = question_prob
        self.terminates_at = terminates_at
        self.randomize_observations = randomize_observations
        self.num_total_questions = num_total_questions
        self.question_interval = question_interval
        self.include_walls_in_observations = include_walls_in_observations
        self.deterministic_objects = deterministic_objects

        if self.deterministic_objects:
            assert (
                self.randomize_observations == "none"
            ), "Deterministic objects should not have randomized observations."

        assert self.num_total_questions % (self.terminates_at + 1) == 0, (
            f"The total number of questions must be a multiple of "
            f"{self.terminates_at + 1}, but it's {self.num_total_questions}"
        )
        assert (self.terminates_at + 1) % self.question_interval == 0, (
            f"The total number of steps must be a multiple of "
            f"{self.question_interval}, but it's {self.terminates_at + 1}"
        )

        self.num_questions_step = (
            self.num_total_questions
            // (self.terminates_at + 1)
            * self.question_interval
        )
        self.total_maximum_episode_rewards = num_total_questions

        self._create_rooms()
        self._compute_room_map()
        self._create_objects()
        self._create_relations_and_objects_for_nn()

        # placeholders for gym spaces
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

        self.CORRECT, self.WRONG, self.PARTIAL = (
            rewards["correct"],
            rewards["wrong"],
            rewards["partial"],
        )
        self.make_everything_static = make_everything_static

        self._init_lists()

    def _init_lists(self) -> None:
        self.hidden_global_states_all = []
        self.observations_all = []
        self.answers_all = []
        self.info_all = []

    def _create_rooms(self) -> None:
        """
        Create room objects based on the room configuration.

        Populates self.rooms with Room objects.
        """
        self.rooms = {}
        for name, config_ in self.room_config.items():
            self.rooms[name] = Room(name, **config_)

    def _create_objects(self) -> None:
        """
        Create all objects (static, independent, dependent, agent) based on configuration.

        Populates self.objects with various object instances.

        Raises:
            AssertionError: If the sum of question probabilities exceeds 1.
        """
        self.objects = {"static": [], "independent": [], "dependent": [], "agent": []}

        for name, init_probs in self.object_init_config["static"].items():
            self.objects["static"].append(
                StaticObject(
                    name,
                    init_probs,
                    self.object_transition_config["static"][name],
                    self.object_question_probs["static"][name],
                    self.deterministic_objects,
                )
            )

        for name, init_probs in self.object_init_config["independent"].items():
            self.objects["independent"].append(
                IndepdentObject(
                    name,
                    init_probs,
                    self.object_transition_config["independent"][name],
                    self.rooms,
                    self.object_question_probs["independent"][name],
                    self.deterministic_objects,
                )
            )

        for name, init_probs in self.object_init_config["dependent"].items():
            self.objects["dependent"].append(
                DependentObject(
                    name,
                    init_probs,
                    self.object_transition_config["dependent"][name],
                    self.objects["independent"],
                    self.object_question_probs["dependent"][name],
                    self.deterministic_objects,
                )
            )

        for name, init_probs in self.object_init_config["agent"].items():
            self.objects["agent"].append(
                Agent(
                    name,
                    init_probs,
                    self.object_transition_config["agent"][name],
                    self.rooms,
                    self.object_question_probs["agent"][name],
                )
            )

        # sanity-check sum of question probs
        question_probs = []
        for obj_type, objects in self.objects.items():
            for obj in objects:
                question_probs.append(obj.question_prob)
        assert abs(sum(question_probs) - 1) <= EPSILON, (
            "The sum of the question probabilities must be <= 1. but it's "
            f"{sum(question_probs)}"
        )

    def _create_relations_and_objects_for_nn(self) -> None:
        """
        Create dictionaries of relations and entities for use in neural networks.

        Populates self.relations and self.entities.
        """
        self.relations = ["north", "east", "south", "west", "at_location"]

        self.entities = {
            "static": [],
            "independent": [],
            "dependent": [],
            "agent": [],
            "room": [],
            "others": ["wall"],
        }
        for obj_type, objects in self.objects.items():
            for obj in objects:
                self.entities[obj_type].append(obj.name)

        for room_name in self.rooms:
            self.entities["room"].append(room_name)

    def _compute_room_map(self) -> None:
        """
        Compute the room layout map as a list of triples.

        Populates self.room_layout.
        """
        self.room_layout = []
        for name, room in self.rooms.items():
            self.room_layout.append([name, "north", room.north])
            self.room_layout.append([name, "east", room.east])
            self.room_layout.append([name, "south", room.south])
            self.room_layout.append([name, "west", room.west])

    def return_room_layout(self, exclude_walls: bool = False) -> list[list[str]]:
        """
        Return the room layout as a list of [room, direction, connected_room] triples.

        Args:
            exclude_walls: If True, exclude connections to walls.

        Returns:
            List of triples representing room connections.
        """
        room_layout = []
        for triple in self.room_layout:
            if exclude_walls:
                if triple[2] != "wall":
                    room_layout.append(triple)
            else:
                room_layout.append(triple)
        return room_layout

    def _compute_hidden_global_state(self) -> None:
        """
        Compute the hidden global state as [head, relation, tail] triples.

        Updates self.hidden_global_state and appends to self.hidden_global_states_all.
        """
        self.hidden_global_state = []

        # room adjacency
        for name, room in self.rooms.items():
            self.hidden_global_state.append([name, "north", room.north])
            self.hidden_global_state.append([name, "east", room.east])
            self.hidden_global_state.append([name, "south", room.south])
            self.hidden_global_state.append([name, "west", room.west])

        # object locations
        for obj_type in ["agent", "static", "independent", "dependent"]:
            for obj in self.objects[obj_type]:
                self.hidden_global_state.append([obj.name, "at_location", obj.location])

        self.hidden_global_states_all.append(deepcopy(self.hidden_global_state))

    def _find_object_by_string(self, obj_str: str) -> Object:
        """
        Find an object by its name.

        Args:
            obj_str: Name of the object to find.

        Returns:
            Object instance if found, None otherwise.
        """
        for obj_type, objects in self.objects.items():
            for obj in objects:
                if obj.name == obj_str:
                    return obj
        return None

    def get_observations_and_question(self, generate_questions: bool = True) -> dict:
        """
        Get observations about the current room and generate questions if required.

        Args:
            generate_questions: Whether to generate questions.

        Returns:
            Dictionary containing observations and questions.
        """
        agent_location = self.objects["agent"][0].location
        self._compute_hidden_global_state()
        self.observations_room = []

        for triple in self.hidden_global_state:
            subj, rel, obj_ = triple
            # show adjacency if subj == agent_location
            if rel in ["north", "east", "south", "west"] and subj == agent_location:
                if self.include_walls_in_observations or (obj_ != "wall"):
                    self.observations_room.append(triple)
            elif rel == "at_location":
                if obj_ == agent_location:  # show objects in same room as agent
                    self.observations_room.append(triple)

        # possibly shuffle the object portion
        if self.randomize_observations == "all":
            random.shuffle(self.observations_room)
        elif self.randomize_observations == "objects":
            first = [
                obs
                for obs in self.observations_room
                if obs[1] in ["north", "east", "south", "west"] or obs[0] == "agent"
            ]
            second = [obs for obs in self.observations_room if obs not in first]
            random.shuffle(second)
            self.observations_room = first + second

        elif self.randomize_observations == "objects_middle":
            first = [
                obs
                for obs in self.observations_room
                if obs[1] in ["north", "east", "south", "west"]
            ]
            third = [obs for obs in self.observations_room if obs[0] == "agent"]
            second = [obs for obs in self.observations_room if obs not in first + third]
            random.shuffle(second)
            self.observations_room = first + second + third
        elif self.randomize_observations == "none":
            pass

        # set up possible questions
        self.questions = []
        self.answers = []

        if generate_questions:
            question_candidates = [
                (obj.name, obj.question_prob)
                for obj_type, objs in self.objects.items()
                for obj in objs
            ]
            names, probs = zip(*question_candidates)
            chosen_tuples = random.choices(
                question_candidates, weights=probs, k=self.num_questions_step
            )
            names = [ct[0] for ct in chosen_tuples]

            objs_chosen = [self._find_object_by_string(name) for name in names]
            for i in range(self.num_questions_step):
                if random.random() > self.question_prob:
                    self.questions.append(None)
                    self.answers.append(None)
                else:
                    obj_chosen = objs_chosen[i]
                    self.questions.append([obj_chosen.name, "at_location", "?"])
                    answer = {"current": obj_chosen.location, "previous": None}

                    for prev_loc in reversed(obj_chosen.history):
                        if prev_loc != obj_chosen.location:
                            answer["previous"] = prev_loc
                            break
                    self.answers.append(answer)

        observations = deepcopy(
            {"room": self.observations_room, "questions": self.questions}
        )
        self.observations_all.append(observations)
        self.answers_all.append(self.answers)

        return observations

    def reset(self) -> tuple[dict, dict]:
        """
        Reset the environment to an initial state.

        Returns:
            Tuple of (observations, info_dict).
        """
        self._init_lists()
        info = {}
        self._create_rooms()
        self._create_objects()
        self.current_time = 0
        self.info_all.append(info)

        # initialize object histories
        for obj_type, objs in self.objects.items():
            for obj in objs:
                obj._update_history()

        # produce observation
        if (self.current_time + 1) % self.question_interval == 0:
            return self.get_observations_and_question(generate_questions=True), info
        else:
            return self.get_observations_and_question(generate_questions=False), info

    def step(
        self, actions: tuple[list[str], str]
    ) -> tuple[dict, int, bool, bool, dict]:
        """
        Take a step in the environment based on provided actions.

        Args:
            actions: Tuple of (question_answers, exploration_action).

        Returns:
            Tuple of (observations, rewards, done, truncated, info_dict).
        """
        actions_qa, action_explore = actions
        assert isinstance(actions_qa, list), "actions_qa must be a list."
        assert isinstance(action_explore, str), "action_explore must be a string."

        # scoring answers
        if len(self.answers) == 0:
            assert actions_qa == [], "You shouldn't answer any questions now"
            rewards = []
        else:
            assert len(actions_qa) == len(self.answers), "You must answer all questions"
            rewards = []
            for user_answer, true_answer in zip(actions_qa, self.answers):
                if user_answer == true_answer["current"]:
                    rewards.append(self.CORRECT)
                elif user_answer == true_answer["previous"]:
                    rewards.append(self.PARTIAL)
                else:
                    rewards.append(self.WRONG)

        # move environment forward
        if not self.make_everything_static:
            for obj in self.objects["independent"]:
                obj.move()
            for obj in self.objects["dependent"]:
                obj.attach()

        # agent action
        self.objects["agent"][0].move(action_explore)

        done = self.current_time == self.terminates_at
        truncated = False
        info = deepcopy({"answers": self.answers, "timestamp": self.current_time})
        self.info_all.append(info)

        self.current_time += 1

        for obj_type, objs in self.objects.items():
            for obj in objs:
                obj._update_history()

        # produce next observation
        if (self.current_time + 1) % self.question_interval == 0:
            obs = self.get_observations_and_question(generate_questions=True)
        else:
            obs = self.get_observations_and_question(generate_questions=False)

        return obs, rewards, done, truncated, info

    def _find_objects_in_room(self, room_name: str) -> dict[str, list]:
        """
        Find all objects currently in a specified room.

        Args:
            room_name: Name of the room to search.

        Returns:
            Dictionary mapping object types to lists of object names.
        """
        objects_in_room = {k: [] for k in self.objects.keys()}
        for obj_type, objs in self.objects.items():
            for obj in objs:
                if obj.location == room_name:
                    objects_in_room[obj_type].append(obj.name)
        return objects_in_room

    def render(
        self,
        render_mode: str = "console",
        figsize: tuple[int, int] = (15, 15),
        cell_text_size: int = 10,
        save_fig_dir: str = None,
        image_format: str = "png",
    ) -> None:
        """
        Render the current state of the environment.

        Args:
            render_mode: How to render the environment ('console' or 'image').
            figsize: Size of the figure when rendering as an image.
            cell_text_size: Text size in cells when rendering as an image.
            save_fig_dir: Directory to save figures if not None.
            image_format: Format to save images in.
        """
        if render_mode == "console":
            pprint(self.hidden_global_state)
        elif render_mode == "image":
            if self.is_notebook:
                clear_output(True)
            plt.figure(figsize=figsize)
            num_rows = len(self.grid)
            num_cols = len(self.grid[0])

            plt.subplot(111)
            plt.title(f"Hidden state at time={self.current_time}")

            for row in range(num_rows):
                for col in range(num_cols):
                    text = ""
                    cell_content = self.grid[row][col]
                    if cell_content != 0:
                        color = "white"
                        room_index = self.room_indexes.index([row, col])
                        room_name = self.names["room"][room_index]
                        text += f"room name={room_name}"

                        objects_in_room = self._find_objects_in_room(room_name)
                        for obj_type, objects in objects_in_room.items():
                            if objects:
                                text += f"\n{obj_type} objects: {objects}"
                    else:
                        color = "black"
                    plt.gca().add_patch(
                        plt.Rectangle((col, num_rows - 1 - row), 1, 1, facecolor=color)
                    )
                    plt.text(
                        col + 0.5,
                        num_rows - 1 - row + 0.5,
                        text,
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=cell_text_size,
                    )
            plt.gca().set_aspect("equal")
            plt.gca().set_xticks(range(num_cols + 1))
            plt.gca().set_yticks(range(num_rows + 1))
            plt.gca().grid(which="both")

            if save_fig_dir is not None:
                os.makedirs(save_fig_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(
                        save_fig_dir,
                        f"hidden_state-{str(self.current_time).zfill(3)}.{image_format}",
                    )
                )
            plt.show()

    def get_rdf_graph_from_state(self, state=None) -> Graph:
        """
        Build an rdflib.Graph from a list of [subject, predicate, object] triples.

        Args:
            state: List of triples to convert. If None, uses current
            hidden_global_state.

        Returns:
            rdflib.Graph representing the state.
        """
        if state is None:
            state = self.hidden_global_state

        g = Graph()

        for s, p, o in state:
            # Convert strings directly to URIRefs without namespace
            g.add((URIRef(s), URIRef(p), URIRef(o)))

        return g

    def get_rdf_graph_from_observations(self) -> Graph:
        """
        Build an rdflib.Graph from the current observations.

        Returns:
            rdflib.Graph representing the current observations.
        """
        g = Graph()

        # Add room observations
        for s, p, o in self.observations_room:
            g.add((URIRef(s), URIRef(p), URIRef(o)))

        return g

    def get_current_observations_rdf_dict(self) -> dict:
        """
        Return a Python dict representing the current observations as RDF.

        Returns:
            Dictionary representation of the current observations as RDF.
        """
        g = self.get_rdf_graph_from_observations()
        return rdf_to_dict(g)

    def get_current_hidden_state_rdf_dict(self) -> dict:
        """
        Return a Python dict representing the current hidden_global_state as RDF.

        Returns:
            Dictionary representation of the current RDF state.
        """
        g = self.get_rdf_graph_from_state(self.hidden_global_state)
        return rdf_to_dict(g)

    def get_all_rdf_dicts(self) -> list[dict]:
        """
        Return a list of Python dicts, one for each time-step's hidden_global_state.

        Returns:
            List of dictionary representations of all historical RDF states.
        """
        all_rdf_dicts = []
        for state in self.hidden_global_states_all:
            g = self.get_rdf_graph_from_state(state)
            all_rdf_dicts.append(rdf_to_dict(g))
        return all_rdf_dicts
