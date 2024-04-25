"""RoomEnv0 environment compatible with gym."""

import logging
import os
import random
from itertools import cycle

import gymnasium as gym

from ..utils import read_json_prod as read_json
from ..utils import read_lines, split_by_possessive

CORRECT = 1
WRONG = 0

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RoomEnv0(gym.Env):
    """The Room environment version 0.

    In this big room, N_{agents} move around and observe N_{humans} placing objects.
    Every time the agents move around (i.e., take a step), they observe one human_{i}
    placing an object somewhere. Each agent can only observe one human at a time.
    Every time the agents takes a step, the environment also asks them where an object
    is. +1 reward is given when it gets right and 0 when it gets wrong.

    This environment is challenging in two ways:

    (1) An agent can't observe the entire room. It can only observe one human at a time.
    This means that the environment is only partially observable. This constraint means
    that it's more beneficial if more than one agent can collaborate. This constraint
    also means that the agents should have a memory system to remember the past
    observations.

    (2) The room environment changes every time. The humans can switch their locations,
    change their objects, and place them at different locations. These changes are
    not completely random. A decent portion of them come from commmon-sense knowledge.
    This means that an agent with both episodic and semantic memory systems will perform
    better than an agent with only one memory system.

    Every string value is lower-cased to avoid confusion!!!

    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        room_size: str = "small",
        probs: dict = {
            "commonsense": 0.7,
            "new_location": 0.1,
            "new_object": 0.1,
            "switch_person": 0.5,
        },
        limits: dict = {
            "heads": None,
            "tails": None,
            "names": None,
            "allow_spaces": True,
        },
        max_step: int = 1000,
        disjoint_entities: bool = True,
        num_agents: int = 1,
    ) -> None:
        """Initialize the environment.

        Args:
            room_size: small or big
            probs: the probabilities that govern the room environment changes.
            limits: Limitation on the triples.
            max_step: maximum step an agent can take. The environment will terminate when
                the number reaches this value.
            disjoint_entities: Assure that the heads and the tails don't overlap.
            num_agents: number of agents in the room.

        """
        super().__init__()
        self.room_size = room_size
        if self.room_size.lower() == "small":
            semantic_knowledge_path = "./data/semantic-knowledge-small.json"
            names_path = "./data/top-human-names-small"

        elif self.room_size.lower() in ["large", "big"]:
            semantic_knowledge_path = "./data/semantic-knowledge.json"
            names_path = "./data/top-human-names"

        logging.debug("Creating an Observation-Question-Answer generator object ...")
        self.limits = limits

        if set(list(self.limits.values())) != {None}:
            logging.warning(f"The obserations will be limited by {self.limits}")

        (
            self.semantic_knowledge,
            self.heads,
            self.relations,
            self.tails,
        ) = self.load_semantic_knowledge(
            semantic_knowledge_path,
            limit_heads=self.limits["heads"],
            limit_tails=self.limits["tails"],
            allow_spaces=self.limits["allow_spaces"],
            disjoint_entities=disjoint_entities,
        )

        assert len(self.relations) == 1, "At the moment there is only one relation."

        self.names = self.read_names(
            names_path, self.limits["names"], self.limits["allow_spaces"]
        )

        assert len(self.names) <= len(self.heads)

        if disjoint_entities:
            lhs = len(set(self.relations + self.names + self.heads + self.tails))
            rhs = (
                len(self.relations)
                + len(self.names)
                + len(self.heads)
                + len(self.tails)
            )

            assert lhs == rhs

        self.probs = probs
        self.max_step = max_step
        self.num_agents = num_agents
        # Our state / action space is quite complex. Here we just make a dummy
        # observation space to bypass the sanity check.
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

    def reset(self) -> tuple:
        """Reset the environment.

        This will place N_{humans} humans in the room. Each human only has one object,
        which will be placed by the human in a random location.

        """
        self.current_time = 0
        random.shuffle(self.names)
        random.shuffle(self.heads)
        self.room = []

        for name, head in zip(self.names, self.heads):
            relation = self.relations[0]  # At the moment there is only one relation.
            tail = self.generate_tail(head, relation)
            if tail is not None:
                self.room.append([f"{name}'s {head}", relation, f"{tail}"])

        navigate = [[i for i in range(len(self.room))] for _ in range(self.num_agents)]
        for navigate_ in navigate:
            random.shuffle(navigate_)

        self.navigate = [cycle(navigate_) for navigate_ in navigate]

        observations = self.generate_observations()
        question, self.answer = self.generate_qa()

        info = {}

        return (observations, question), info

    def generate_observations(self) -> list:
        """Generate a random obseration.

        Returns:
            observations: e.g., ["tae's laptop, "atlocation", "tae's desk", 10]
                The last element in the list accounts for the timestamp.

        """
        observations = {
            i: [*self.room[next(navigate_)], self.current_time]
            for i, navigate_ in enumerate(self.navigate)
        }

        return observations

    def generate_qa(self) -> tuple[list, str]:
        """Generate a question and the answer.

        Returns:
            question: e.g., ["tae's laptop", "atlocation", "?", 10]
            answer: e.g., desk

        """
        random_choice = random.choice(self.room)
        question = random_choice[:2] + ["?"] + [self.current_time]
        answer = random_choice[2]

        return question, answer

    def generate_tail(self, head: str, relation: str) -> str:
        """This simulates humans placing their objects in their desired locations.

        Note that "head" shouldn't include a human name.

        """
        relevant_tails = [
            triple[2]
            for triple in self.semantic_knowledge
            if triple[0] == head and triple[1] == relation
        ]
        assert len(relevant_tails) == 1

        if random.random() < self.probs["commonsense"]:
            logging.debug(f"Generating a common location for {head} ...")
            tail = relevant_tails[0]
        else:
            logging.debug(f"Generating a NON common location for {head} ...")
            while True:
                tail = random.choice(self.tails)

                if tail not in relevant_tails:
                    break

        return tail

    def renew(self) -> None:
        """Renew the room.

        This is done every time when the agent takes a step. This is doen to simulate
        that the room changes. People move. They place their objects in different
        locations. They also change their objects. All of these are done in a random
        manner. The randomness can be adjusted from the argument `probs`.

        With the chance of probs["new_location"], an object will be placed at a new
        location.

        When the object is placed at a new locaiton, with the chance of
        probs["commonsense"], an object will be placed at a commonsense-knowledge spot.

        With the chance of probs["new_object"], the person with the object will have
        a new random object.

        With the chance of probs["switch_person"], two persons switch their spots.

        """
        room = []
        for head, relation, tail in self.room:
            name1, head = split_by_possessive(head)

            if random.random() < self.probs["new_object"]:
                while True:
                    new_head = random.choice(self.heads)
                    if new_head != head:
                        head = new_head
                        tail = self.generate_tail(head, relation)
                        break
            else:
                if random.random() < self.probs["new_location"]:
                    while True:
                        new_tail = self.generate_tail(head, relation)
                        if new_tail != tail:
                            tail = new_tail
                            break

            room.append(
                [f"{name1}'s {head}", relation, f"{tail}"],
            )

        if random.random() < self.probs["switch_person"]:
            i, j = random.sample(
                range(
                    0,
                    len(
                        self.room,
                    ),
                ),
                2,
            )
            room[i], room[j] = room[j], room[i]

        self.room = room

    def step(self, action: str) -> tuple[tuple, int, bool, bool, dict]:
        """An agent takes an action.

        Args:
            action: This is the agent's answer to the previous question.

        Returns:
            observations: e.g., ["tae's laptop, "atlocation", "tae's desk", 10]
            question: e.g., ["tae's laptop", "atlocation", "?", 10]
            answer: e.g., desk
            reward: +1 for correct answer, 0 for wrong answer.
            done: True if the current_time reaches the max_step.
            truncated: False

        """
        if str(action).lower() == self.answer.lower():
            logging.info(
                f"The prediction ({action}) matches the answer ({self.answer})!"
            )
            reward = CORRECT

        else:
            logging.info(
                f"The prediction ({action}) does NOT match the answer ({self.answer})!"
            )
            reward = WRONG

        self.current_time += 1

        # Things will change in the room!
        self.renew()

        observations = self.generate_observations()
        question, self.answer = self.generate_qa()

        info = {}

        if self.current_time >= self.max_step:
            done = True
        else:
            done = False

        truncated = False

        return (observations, question), reward, done, truncated, info

    def render(self, mode="console") -> None:
        if mode != "console":
            raise NotImplementedError()
        else:
            pass

    def close(self):
        pass

    @staticmethod
    def read_names(
        path: str = "./data/top-human-names",
        limit_names: int = None,
        allow_spaces: bool = False,
    ) -> list:
        """Read 20 most common names.

        Args:
            path: The path to the top 20 human name list.
            limit_names: Limit the number of names
            allow_spaces: Whether to include words that have spaces
                (e.g., corner of two streets)

        Returns:
            names: human names (e.g., james)

        """
        names = read_lines(path)

        if not allow_spaces:
            names = [name for name in names if len(name.split("_")) == 1]

        if limit_names:
            logging.warning(f"The number of names will be limited to {limit_names}")
            names = sorted(names, key=len)
            names = names[:limit_names]

        logging.info(f"Reading {path} complete! There are {len(names)} names in total")

        return names

    @staticmethod
    def load_semantic_knowledge(
        path: str,
        limit_heads: int = None,
        limit_tails: int = None,
        allow_spaces: bool = False,
        disjoint_entities: bool = True,
    ) -> tuple[list, list, list, list]:
        """Load saved semantic knowledge.

        Args:
            path: the path to the pretrained semantic memory.
            limit_heads: Limit the number of heads (e.g., 10)
            limit_tails: Limit the number of tails per heads (e.g., 1)
            allow_spaces: Whether to include words that have spaces
                (e.g., corner of two streets)
            disjoint_entities: Whether to force that there are no common elements between
                entities.

        Returns:
            semantic_knowledge
            heads
            relations
            tails

        """
        logging.debug(f"loading the semantic knowledge from {path}...")
        semantic_knowledge = read_json(path)

        heads = sorted(list(set(semantic_knowledge.keys())))
        if disjoint_entities:
            logging.warning("Tails that are heads will be removed.")
            semantic_knowledge = {
                key: {
                    key_: [tail for tail in val_ if tail["tail"] not in heads]
                    for key_, val_ in val.items()
                }
                for key, val in semantic_knowledge.items()
            }

        semantic_knowledge = {
            key: {
                key_: val_
                for key_, val_ in val.items()
                if len([tail for tail in val_ if tail["tail"]]) > 0
            }
            for key, val in semantic_knowledge.items()
        }
        semantic_knowledge = {
            key: val for key, val in semantic_knowledge.items() if len(val) > 0
        }
        logging.info("empty entities are removed")

        # sort the semantic knowledge by its highest weight to be sure.
        semantic_knowledge = {
            key: {
                key_: sorted(val_, key=lambda x: -x["weight"])
                for key_, val_ in val.items()
            }
            for key, val in semantic_knowledge.items()
        }

        if not allow_spaces:
            semantic_knowledge = {
                key: {
                    key_: [v for v in val_ if len(v["tail"].split("_")) == 1]
                    for key_, val_ in val.items()
                }
                for key, val in semantic_knowledge.items()
                if (len(key.split("_"))) == 1
            }

        if limit_heads:
            logging.warning(f"Limiting the number of heads to {limit_heads} ...")
            semantic_knowledge = {
                key: val
                for idx, (key, val) in enumerate(semantic_knowledge.items())
                if idx < limit_heads
            }

        if limit_tails:
            logging.warning(
                f"Limiting the number of tails per head to {limit_tails} ..."
            )
            semantic_knowledge = {
                key: {key_: val_[:limit_tails] for key_, val_ in val.items()}
                for key, val in semantic_knowledge.items()
            }

        heads = sorted(list(set(semantic_knowledge.keys())))
        tails = sorted(
            list(
                set(
                    [
                        foo["tail"]
                        for key, val in semantic_knowledge.items()
                        for key_, val_ in val.items()
                        for foo in val_
                    ]
                )
            )
        )
        relations = sorted(
            list(
                set(
                    [
                        key_
                        for key, val in semantic_knowledge.items()
                        for key_, val_ in val.items()
                    ]
                )
            )
        )
        logging.info(f"semantic knowledge successfully loaded from {path}!")

        semantic_knowledge_list = []
        for key, val in semantic_knowledge.items():
            for k, v in val.items():
                semantic_knowledge_list.append([key, k, v[0]["tail"]])

        return semantic_knowledge_list, heads, relations, tails
