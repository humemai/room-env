import unittest

from room_env.memory import EpisodicMemory, Memory, SemanticMemory, ShortMemory


class MemoryTest(unittest.TestCase):
    def test_all(self) -> None:
        for capacity in [1, 2, 4, 8, 16, 32]:
            m = EpisodicMemory(capacity=capacity)
            m = SemanticMemory(capacity=capacity)
            m = ShortMemory(capacity=capacity)
