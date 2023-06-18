"""Room environment compatible with gym.

This env uses the RoomDes (room_env/envs/des.py), and Memory classes.
This is a more generalized version than RoomEnv2.
"""
import logging
import os
import random
from copy import deepcopy
from typing import List, Tuple

import gymnasium as gym

from ..memory import EpisodicMemory, SemanticMemory, ShortMemory
from ..policy import answer_question, encode_observation, manage_memory
from ..utils import seed_everything

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Object:
    def __init__(
        self,
        name: str,
        type: str,
    ) -> None:
        """Entity, e.g., human, object, room.

        Args
        ----
        name: e.g., Tae, laptop, bed
        """
        self.name = name
        self.type = type

    def __repr__(self) -> str:
        return f"Object({self.name}, {self.type})"

    def step(self) -> None:
        """Move object to another room."""
        pass


class StaticObject(Object):
    def __init__(self, name: str, location: str) -> None:
        super().__init__(name, "static")
        self.location = location

    def __repr__(self) -> str:
        return f"StaticObject({self.name}, {self.location})"

    def step(self) -> None:
        """Static object does not move."""
        pass


class IndepdentObject(Object):
    def __init__(self, name: str, probs: dict) -> None:
        super().__init__(name, "independent")
        self.probs = probs
        self.step()

    def __repr__(self) -> str:
        return f"IndependentObject({self.name}, {self.location})"

    def step(self) -> None:
        """Indendent object objects to another room."""
        self.location = random.choices(
            list(self.probs.keys()),
            weights=list(self.probs.values()),
            k=1,
        )[0]


class DependentObject(Object):
    def __init__(
        self, name: str, dependence: List[Tuple[IndepdentObject, float]]
    ) -> None:
        super().__init__(name, "dependent")
        self.dependence = dependence
        self.independent_objects = [io for io, prob in self.dependence]

        while True:
            possible_attachments = []
            for io, prob in self.dependence:
                if random.random() < prob:
                    possible_attachments.append(io)
            if len(possible_attachments) > 0:
                break

        io = random.choice(possible_attachments)
        self.attached = io
        self.location = self.attached.location

    def step(self) -> None:
        """Move together with independent object."""
        possible_attachments = []
        for io in self.independent_objects:
            if io.location == self.location:
                for io_, prob in self.dependence:
                    if io == io_:
                        if random.random() < prob:
                            possible_attachments.append(io)

        if len(possible_attachments) > 0:
            io = random.choice(possible_attachments)
            self.attached = io
            self.location = self.attached.location

        else:
            self.attached = None

    def __repr__(self) -> str:
        return f"DependentObject({self.name}, {self.location}, {self.attached})"


class Agent(Object):
    def __init__(self, init_probs: dict) -> None:
        super().__init__("agent", "agent")
        self.location = random.choices(
            list(init_probs.keys()),
            weights=list(init_probs.values()),
            k=1,
        )[0]

    def __repr__(self) -> str:
        return f"Agent({self.name}, {self.location})"

    def step(self, location: str) -> None:
        """Agent can choose where to go."""
        self.location = location


class RoomEnv2(gym.Env):
    """the Room environment version 2.

    This environment is more formalized than the previous environments. There are three
    policies here.

    1. Question answering policy: $\pi_{qa}(a_{qa}|M_{long})$
    2. Memory management policy: $\pi_{memory}(a_{memory} | M_{short}, M_{long})$
    3. Exploration policy: $\pi_{explore}(a_{explore} | M_{long})$

    The idea is to fix two of them and let RL learn one.

    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        seed: int = 42,
        policies: dict = {
            "question_answer": "episodic_semantic",
            "memory_management": "RL",
            "explore": "one_by_one",
        },
        capacity: dict = {"episodic": 16, "semantic": 16, "short": 1},
        question_prob: int = 1.0,
        total_episode_rewards: int = 100,
        pretrain_semantic: bool = False,
        room_config: dict = None,
    ) -> None:
        """

        Args
        ----
        seed: random seed number
        policies:
            question_answer:
                "RL": Reinforcement learning to learn the policy.
                "episodic_semantic": First look up the episodic and then the semantic.
                "semantic_episodic": First look up the semantic and then the episodic.
                "episodic": Only look up the episodic.
                "semantic": Only look up the semantic.
                "random": Take one of the two actions uniform-randomly.
                "neural": Neural network policy
            memory_management:
                "RL": Reinforcement learning to learn the policy.
                "episodic": Always take action 1: move to the episodic.
                "semantic": Always take action 2: move to the semantic.
                "forget": Always take action 3: forget the oldest short-term memory.
                "random": Take one of the three actions uniform-randomly.
                "neural": Neural network policy
            explore:
                "RL": Reinforcement learning to learn the policy.
                "uniform_random": Choose one of the sub-graphs uniformly randomly.
                "one_by_one": Choose one of the sub-graphs one by one.
                "neural": Neural network policy
        capacity: memory capactiy of the agent.
            e.g., {"episodic": 1, "semantic": 1}
        question_prob: The probability of a question being asked at every observation.
        total_episode_rewards: total episode rewards
        pretrain_semantic: whether to prepopulate the semantic memory with ConceptNet
                           or not
        room_config: room configuration

        """
        self.seed = seed
        seed_everything(self.seed)
        self.policies = policies
        assert (
            len([pol for pol in self.policies.values() if pol.lower() == "rl"]) == 1
        ), "Only one policy can be RL."
        self.capacity = capacity
        self.question_prob = question_prob
        assert 0 < self.question_prob <= 1, "Question probability must be in (0, 1]."
        self.total_episode_rewards = total_episode_rewards
        self.pretrain_semantic = pretrain_semantic
        self.room_config = room_config

        self._populate_rooms()

        # Our state space is quite complex. Here we just make a dummy observation space.
        # to bypass the sanity check.
        self.observation_space = gym.spaces.Discrete(1)

        if self.policies["question_answer"].lower() == "rl":
            # 0 for episodic and 1 for semantic
            self.action_space = gym.spaces.Discrete(2)
        if self.policies["memory_management"].lower() == "rl":
            # 0 for episodic, 1 for semantic, and 2 to forget
            self.action_space = gym.spaces.Discrete(3)
        if self.policies["explore"].lower() == "rl":
            raise NotImplementedError

    def _populate_rooms(self) -> None:
        """Populate the rooms with objects."""
        # static objects
        bed = StaticObject("bed", "bedroom")
        desk = StaticObject("desk", "officeroom")
        table = StaticObject("table", "livingroom")

        # independent objects
        tae = IndepdentObject("tae", {"officeroom": 0.5, "livingroom": 0.5})
        michael = IndepdentObject("michael", {"bedroom": 0.5, "livingroom": 0.5})
        vincent = IndepdentObject("vincent", {"bedroom": 0.5, "officeroom": 0.5})

        # dependent objects
        laptop = DependentObject("laptop", [(tae, 0.7), (michael, 0.1), (vincent, 0.3)])
        phone = DependentObject("phone", [(tae, 0.3), (michael, 0.1), (vincent, 0.7)])
        headset = DependentObject(
            "headset", [(tae, 0.5), (michael, 0.5), (vincent, 0.5)]
        )
        keyboard = DependentObject(
            "keyboard", [(tae, 0.9), (michael, 0.7), (vincent, 0.5)]
        )

        # agent
        self.agent = Agent({"bedroom": 0.333, "officeroom": 0.333, "livingroom": 0.333})

        self.rooms = ["bedroom", "officeroom", "livingroom"]
        self.rooms = {room: [] for room in self.rooms}
        self.static_objects = [bed, desk, table]
        self.independent_objects = [tae, michael, vincent]
        self.dependent_objects = [laptop, phone, headset, keyboard]

        for obj in self.static_objects:
            self.rooms[obj.location].append(obj)

        for obj in self.independent_objects:
            self.rooms[obj.location].append(obj)

        for obj in self.dependent_objects:
            self.rooms[obj.location].append(obj)

        self.rooms[self.agent.location].append(self.agent)

    def init_memory_systems(self) -> None:
        """Initialize the agent's memory systems."""
        self.memory_systems = {
            "episodic": EpisodicMemory(capacity=self.capacity["episodic"]),
            "semantic": SemanticMemory(capacity=self.capacity["semantic"]),
            "short": ShortMemory(capacity=self.capacity["short"]),
        }

        if self.pretrain_semantic:
            assert self.capacity["semantic"] > 0
            _ = self.memory_systems["semantic"].pretrain_semantic(
                self.des.semantic_knowledge,
                return_remaining_space=False,
                freeze=False,
            )

    def generate_sequences(self) -> None:
        """Generate human and question sequences in advance."""
        if self.observation_params.lower() == "perfect":
            if self.allow_random_human:
                self.human_sequence = random.choices(
                    list(self.des.humans), k=self.des.until + 1
                )
            else:
                self.human_sequence = (
                    self.des.humans * (self.des.until // len(self.des.humans) + 1)
                )[: self.des.until + 1]
        else:
            raise NotImplementedError

        if self.allow_random_question:
            self.question_sequence = [
                random.choice(self.human_sequence[: i + 1])
                for i in range(len(self.human_sequence))
            ]
        else:
            self.question_sequence = [self.human_sequence[0]]
            self.des.run()
            assert (
                len(self.des.states)
                == len(self.des.events) + 1
                == len(self.human_sequence)
            )
            for i in range(len(self.human_sequence) - 1):
                start = max(i + 2 - len(self.des.humans), 0)
                end = i + 2
                humans_observed = self.human_sequence[start:end]

                current_state = self.des.states[end - 1]
                humans_not_changed = []
                for j, human in enumerate(humans_observed):
                    observed_state = self.des.states[start + j]

                    is_changed = False
                    for to_check in ["object", "object_location"]:
                        if (
                            current_state[human][to_check]
                            != observed_state[human][to_check]
                        ):
                            is_changed = True
                    if not is_changed:
                        humans_not_changed.append(human)

                self.question_sequence.append(random.choice(humans_not_changed))

            self.des._initialize()

        effective_question_sequence = []
        for i, question in enumerate(self.question_sequence[:-1]):
            if random.random() < self.question_prob:
                effective_question_sequence.append(question)
            else:
                effective_question_sequence.append(None)
        # The last observation shouldn't have a question
        effective_question_sequence.append(None)
        self.question_sequence = effective_question_sequence

        assert len(self.human_sequence) == len(self.question_sequence)

        self.num_questions = sum(
            [True for question in self.question_sequence if question is not None]
        )
        if self.varying_rewards:
            self.CORRECT = self.total_episode_rewards / self.num_questions
            self.WRONG = -self.CORRECT
        else:
            self.CORRECT = 1
            self.WRONG = -1

    @staticmethod
    def extract_memory_entries(memory_systems: dict) -> dict:
        """Extract the entries from the Memory objects.
        Ars
        ---
        memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                        "short": ShortMemory}

        Returns
        -------
        memory_systems_: memory_systems only with entries.
        """
        memory_systems_ = {}
        for key, value in memory_systems.items():
            memory_systems_[key] = deepcopy(value.entries)

        return memory_systems_

    def generate_oqa(
        self, increment_des: bool = False
    ) -> Tuple[dict, dict, dict, bool]:
        """Generate an observation, question, and answer.

        Args
        ----
        increment_des: whether or not to take a step in the DES.

        Returns
        -------
        observation = {
            "human": <human>,
            "object": <obj>,
            "object_location": <obj_loc>,
        }
        question = {"human": <human>, "object": <obj>}
        answer = <obj_loc>
        is_last: True, if its the last observation in the queue, othewise False

        """
        human_o = self.human_sequence.pop(0)
        human_q = self.question_sequence.pop(0)

        is_last_o = len(self.human_sequence) == 0
        is_last_q = len(self.question_sequence) == 0

        assert is_last_o == is_last_q
        is_last = is_last_o

        if increment_des:
            self.des.step()

        obj_o = self.des.state[human_o]["object"]
        obj_loc_o = self.des.state[human_o]["object_location"]
        observation = deepcopy(
            {
                "human": human_o,
                "object": obj_o,
                "object_location": obj_loc_o,
                "current_time": self.des.current_time,
            }
        )

        if human_q is not None:
            obj_q = self.des.state[human_q]["object"]
            obj_loc_q = self.des.state[human_q]["object_location"]

            question = deepcopy({"human": human_q, "object": obj_q})
            answer = deepcopy(obj_loc_q)

        else:
            question = None
            answer = None

        return observation, question, answer, is_last

    def reset(self) -> dict:
        """Reset the environment.


        Returns
        -------
        state

        """
        self.des._initialize()
        self.generate_sequences()
        self.init_memory_systems()
        info = {}
        self.obs, self.question, self.answer, self.is_last = self.generate_oqa(
            increment_des=False
        )

        if self.policies["encoding"].lower() == "rl":
            return deepcopy(self.obs), info

        if self.policies["memory_management"].lower() == "rl":
            encode_observation(self.memory_systems, self.policies["encoding"], self.obs)
            return deepcopy(self.extract_memory_entries(self.memory_systems)), info

        if self.policies["question_answer"].lower() == "rl":
            encode_observation(self.memory_systems, self.policies["encoding"], self.obs)
            manage_memory(self.memory_systems, self.policies["memory_management"])
            while True:
                if (self.question is None) and (self.answer is None):
                    (
                        self.obs,
                        self.question,
                        self.answer,
                        self.is_last,
                    ) = self.generate_oqa(increment_des=True)
                    encode_observation(
                        self.memory_systems, self.policies["encoding"], self.obs
                    )
                    manage_memory(
                        self.memory_systems, self.policies["memory_management"]
                    )
                else:
                    return {
                        "memory_systems": deepcopy(
                            self.extract_memory_entries(self.memory_systems)
                        ),
                        "question": deepcopy(self.question),
                    }, info

        raise ValueError

    def step(self, action: int) -> Tuple[Tuple, int, bool, bool, dict]:
        """An agent takes an action.

        Args
        ----
        action: This depends on the state

        Returns
        -------
        state, reward, done, truncated, info

        """
        info = {}
        truncated = False
        if self.policies["encoding"].lower() == "rl":
            # This is a dummy code
            self.obs = self.obs[action]
            encode_observation(self.memory_systems, self.policies["encoding"], self.obs)
            manage_memory(self.memory_systems, self.policies["memory_management"])

            if (self.question is None) and (self.answer is None):
                reward = 0
            else:
                pred = answer_question(
                    self.memory_systems, self.policies["question_answer"], self.question
                )
                if str(pred).lower() == self.answer:
                    reward = self.CORRECT
                else:
                    reward = self.WRONG
            self.obs, self.question, self.answer, self.is_last = self.generate_oqa(
                increment_des=True
            )
            state = deepcopy(self.obs)

            if self.is_last:
                done = True
            else:
                done = False

            return state, reward, done, truncated, info

        if self.policies["memory_management"].lower() == "rl":
            if action == 0:
                manage_memory(self.memory_systems, "episodic")
            elif action == 1:
                manage_memory(self.memory_systems, "semantic")
            elif action == 2:
                manage_memory(self.memory_systems, "forget")
            else:
                raise ValueError

            if (self.question is None) and (self.answer is None):
                reward = 0
            else:
                pred = answer_question(
                    self.memory_systems, self.policies["question_answer"], self.question
                )
                if str(pred).lower() == self.answer:
                    reward = self.CORRECT
                else:
                    reward = self.WRONG

            self.obs, self.question, self.answer, self.is_last = self.generate_oqa(
                increment_des=True
            )
            encode_observation(self.memory_systems, self.policies["encoding"], self.obs)
            state = deepcopy(self.extract_memory_entries(self.memory_systems))

            if self.is_last:
                done = True
            else:
                done = False

            return state, reward, done, truncated, info

        if self.policies["question_answer"].lower() == "rl":
            if action == 0:
                pred = answer_question(self.memory_systems, "episodic", self.question)
            elif action == 1:
                pred = answer_question(self.memory_systems, "semantic", self.question)
            else:
                raise ValueError

            if str(pred).lower() == self.answer:
                reward = self.CORRECT
            else:
                reward = self.WRONG

            while True:
                (
                    self.obs,
                    self.question,
                    self.answer,
                    self.is_last,
                ) = self.generate_oqa(increment_des=True)
                encode_observation(
                    self.memory_systems, self.policies["encoding"], self.obs
                )
                manage_memory(self.memory_systems, self.policies["memory_management"])

                if self.is_last:
                    state = None
                    done = True
                    return state, reward, done, truncated, info
                else:
                    done = False

                if (self.question is not None) and (self.answer is not None):
                    state = {
                        "memory_systems": deepcopy(
                            self.extract_memory_entries(self.memory_systems)
                        ),
                        "question": deepcopy(self.question),
                    }

                    return state, reward, done, truncated, info

    def render(self, mode="console") -> None:
        if mode != "console":
            raise NotImplementedError()
        else:
            pass
