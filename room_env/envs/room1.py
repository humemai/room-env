"""Room environment compatible with gym.

This env uses the RoomDes (room_env/envs/des.py), and Memory classes.
This is a more generalized version than RoomEnv0.
"""
import logging
import os
import random
from copy import deepcopy
from typing import Tuple

import gymnasium as gym

from ..des import RoomDes
from ..memory import EpisodicMemory, SemanticMemory, ShortMemory
from ..policy import answer_question, encode_observation, manage_memory
from ..utils import seed_everything

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RoomEnv1(gym.Env):
    """The Room environment version 1.

    This env includes three state-action spaces. You have to choose which one of the
    three will be RL trained.

    Memory management.
        State: episodic, semantic, and short-term memory systems at time t
        Action: (0) Move the oldest short-term memory to the episodic,
                (1) to the semantic, or (2) forget it

    Question-answer
        State: episodic and semantic memory systems at time t
        Action: (0) Select the episodic memory system to answer the question, or
                (1) the semantic

    Encoding an observation to a short-term memory. The state space is
        (i) triple-based, (ii) text-based, or (iii) image-based.
        Triple
            State: [(head_i, relation_i, tail_i) | i is from 1 to N]
            Action: Choose one of the N triples (actions) to be encoded as
                    a short-term memory.
        Text
            State: [token_1, token_2, …, token_N]
            Action: This is actually now N^3, where the first, second and third are to
                    choose head, relation, and tail, respectively.
        Image
            State: An image with objects
            Action: Not sure yet …

    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        des_size: str = "l",
        seed: int = 42,
        policies: dict = {
            "memory_management": "RL",
            "question_answer": "episodic_semantic",
            "encoding": "argmax",
        },
        capacity: dict = {"episodic": 16, "semantic": 16, "short": 1},
        question_prob: int = 1.0,
        observation_params: str = "perfect",
        allow_random_human: bool = False,
        allow_random_question: bool = False,
        total_episode_rewards: int = 128,
        pretrain_semantic: bool = False,
        check_resources: bool = True,
        varying_rewards: bool = False,
    ) -> None:
        """

        Args
        ----
        des_size: "xxs", "xs", "s", "m", or "l".
        seed: random seed number
        policies:
            memory_management:
                "RL": Reinforcement learning to learn the policy.
                "episodic": Always take action 1: move to the episodic.
                "semantic": Always take action 2: move to the semantic.
                "forget": Always take action 3: forget the oldest short-term memory.
                "random": Take one of the three actions uniform-randomly.
                "neural": Neural network policy
            question_answer:
                "RL": Reinforcement learning to learn the policy.
                "episodic_semantic": First look up the episodic and then the semantic.
                "semantic_episodic": First look up the semantic and then the episodic.
                "episodic": Only look up the episodic.
                "semantic": Only look up the semantic.
                "random": Take one of the two actions uniform-randomly.
                "neural": Neural network policy
            encoding:
                "RL": Reinforcement learning to learn the policy.
                "argmax": Take the triple with the highest score.
                "neural": Neural network policy
        capacity: memory capactiy of the agent.
            e.g., {"episodic": 1, "semantic": 1}
        question_prob: The probability of a question being asked at every observation.
        observation_params: At the moment this is only "perfect".
        allow_random_human: whether or not to generate a random human sequence.
        allow_random_question: whether or not to geneate a random question sequence.
        total_episode_rewards: total episode rewards
        pretrain_semantic: whether to prepopulate the semantic memory with ConceptNet
                           or not
        check_resources: whether to check the resources in the DES.
        varying_rewards: If true, then the rewards are scaled in every episode so that
             total_episode_rewards is total_episode_rewards.

        """
        self.seed = seed
        seed_everything(self.seed)
        self.policies = policies
        assert len([pol for pol in self.policies.values() if pol.lower() == "rl"]) == 1
        self.capacity = capacity
        self.question_prob = question_prob

        self.observation_params = observation_params

        self.allow_random_human = allow_random_human
        self.allow_random_question = allow_random_question
        self.total_episode_rewards = total_episode_rewards
        self.pretrain_semantic = pretrain_semantic
        self.check_resources = check_resources
        self.varying_rewards = varying_rewards

        # Our state space is quite complex. Here we just make a dummy observation space.
        # to bypass the sanity check.
        self.observation_space = gym.spaces.Discrete(1)

        if self.policies["memory_management"].lower() == "rl":
            # 0 for episodic, 1 for semantic, and 2 to forget
            self.action_space = gym.spaces.Discrete(3)
        if self.policies["question_answer"].lower() == "rl":
            # 0 for episodic and 1 for semantic
            self.action_space = gym.spaces.Discrete(2)
        if self.policies["encoding"].lower() == "rl":
            raise NotImplementedError

        self.des_size = des_size
        self.des = RoomDes(
            des_size=self.des_size,
            check_resources=self.check_resources,
        )
        assert 0 < self.question_prob <= 1

        self.init_memory_systems()

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
    def extract_memory_entires(memory_systems: dict) -> dict:
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
            return deepcopy(self.extract_memory_entires(self.memory_systems)), info

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
                            self.extract_memory_entires(self.memory_systems)
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
            state = deepcopy(self.extract_memory_entires(self.memory_systems))

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
                            self.extract_memory_entires(self.memory_systems)
                        ),
                        "question": deepcopy(self.question),
                    }

                    return state, reward, done, truncated, info

    def render(self, mode="console") -> None:
        if mode != "console":
            raise NotImplementedError()
        else:
            pass
