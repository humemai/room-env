"""RoomEnv1 environment compatible with gym.

This env uses the RoomDes (room_env/envs/des.py).
This is a more generalized version than RoomEnv0.
"""

import logging
import os
import random

import gymnasium as gym

from ..des import RoomDes
from ..utils import seed_everything

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RoomEnv1(gym.Env):
    """The Room environment version 1.
    self.question_sequence
        Every string value is lower-cased to avoid confusion!!!
    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        des_size: str = "l",
        seed: int = 42,
        question_prob: int = 1.0,
        allow_random_human: bool = False,
        allow_random_question: bool = False,
        check_resources: bool = True,
    ) -> None:
        """

        Args:
            des_size: "xxs", "xs", "s", "m", or "l".
            seed: random seed number
            question_prob: The probability of a question being asked at every observation.
            allow_random_human: whether or not to generate a random human sequence.
            allow_random_question: whether or not to geneate a random question sequence.
            check_resources: whether to check the resources in the DES.

        """
        super().__init__()
        self.seed = seed
        seed_everything(self.seed)
        self.question_prob = question_prob
        self.allow_random_human = allow_random_human
        self.allow_random_question = allow_random_question
        self.check_resources = check_resources

        # Our state / actionspace is quite complex. Here we just make a dummy spaces
        # to bypass the gymnasium sanity check.
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

        self.des_size = des_size
        self.des = RoomDes(
            des_size=self.des_size,
            check_resources=self.check_resources,
        )
        self.total_maximum_episode_rewards = self.des.until
        assert 0 < self.question_prob <= 1

    def generate_sequences(self) -> None:
        """Generate human and question sequences in advance."""
        if self.allow_random_human:
            self.human_sequence = random.choices(
                list(self.des.humans), k=self.des.until + 1
            )
        else:
            self.human_sequence = (
                self.des.humans * (self.des.until // len(self.des.humans) + 1)
            )[: self.des.until + 1]

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

        self.CORRECT = 1
        self.WRONG = -1

    def generate_oqa(
        self, increment_des: bool = False
    ) -> tuple[dict, dict, dict, bool]:
        """Generate an observation, question, and answer.

        Args:
            increment_des: whether or not to take a step in the DES.

        Returns:
            observation: [head, relation, tail, timestamp]
            question: [head, relation, ?, timestamp]
            answer = tail
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
        observation = [
            f"{human_o}'s {obj_o}",
            "atlocation",
            obj_loc_o,
            self.des.current_time,
        ]
        if human_q is not None:
            obj_q = self.des.state[human_q]["object"]
            obj_loc_q = self.des.state[human_q]["object_location"]

            question = [
                f"{human_q}'s {obj_q}",
                "atlocation",
                "?",
                self.des.current_time,
            ]
            answer = obj_loc_q

        else:
            question = None
            answer = None

        return observation, question, answer, is_last

    def reset(self) -> tuple[dict, dict]:
        """Reset the environment.

        This method somehow can't take arguments. I think it's a bug.


        Returns:
            state, info

        """
        self.des._initialize()
        self.generate_sequences()
        info = {}
        self.obs, self.question, self.answer, self.is_last = self.generate_oqa(
            increment_des=False
        )

        return (self.obs, self.question), info

    def step(self, action: str) -> tuple[tuple, int, bool, bool, dict]:
        """An agent takes an action.

        Args:
            action: An answer to the question.

        Returns:
            state, reward, done, truncated, info

        """
        info = {}
        truncated = False
        if (self.question is None) and (self.answer is None):
            reward = 0
        else:
            if action.lower() == self.answer:
                reward = self.CORRECT
            else:
                reward = self.WRONG

        self.obs, self.question, self.answer, self.is_last = self.generate_oqa(
            increment_des=True
        )

        if self.is_last:
            done = True
        else:
            done = False

        return (self.obs, self.question), reward, done, truncated, info

    def render(self, mode="console") -> None:
        if mode != "console":
            raise NotImplementedError()
        else:
            pass
