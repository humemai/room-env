import unittest

from room_env.memory import EpisodicMemory, Memory, SemanticMemory, ShortMemory
from room_env.policy import answer_question, encode_observation, manage_memory


class PolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory_systems = {
            "short": ShortMemory(capacity=1),
            "episodic": EpisodicMemory(capacity=16),
            "semantic": SemanticMemory(capacity=16),
        }

    def test_encode_observation(self):
        with self.assertRaises(NotImplementedError):
            encode_observation(
                memory_systems=self.memory_systems,
                policy="foo",
                obs={"human": "Tae", "object": "Ubuntu", "object_location": "Linux"},
            )

    def test_manage_memory(self):
        with self.assertRaises(NotImplementedError):
            manage_memory(memory_systems=self.memory_systems, policy="neural")

    def test_answer_question(self):
        with self.assertRaises(NotImplementedError):
            answer_question(
                memory_systems=self.memory_systems,
                policy="neural",
                question={"human": "Tae", "object": "Ubuntu"},
            )
