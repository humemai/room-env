import logging
import random
import unittest

import gymnasium as gym

import room_env

logger = logging.getLogger()
logger.disabled = True


class RoomEnv1Test(unittest.TestCase):
    def test_all(self) -> None:
        for des_size in ["xxs", "xs", "s", "m", "l"]:
            for question_prob in [0.1, 0.2, 0.4, 1]:
                for allow_random_human in [True, False]:
                    for allow_random_question in [True, False]:
                        for pretrain_semantic in [True, False]:
                            for check_resources in [True, False]:
                                for varying_rewards in [True, False]:
                                    env = gym.make(
                                        "RoomEnv-v1",
                                        des_size=des_size,
                                        question_prob=question_prob,
                                        allow_random_human=allow_random_human,
                                        allow_random_question=allow_random_question,
                                        pretrain_semantic=pretrain_semantic,
                                        check_resources=check_resources,
                                        varying_rewards=varying_rewards,
                                    )
                                    state, info = env.reset()
                                    while True:
                                        (
                                            state,
                                            reward,
                                            done,
                                            truncated,
                                            info,
                                        ) = env.step(random.randint(0, 2))
                                        if done:
                                            break

    def test_wrong_init0(self) -> None:

        with self.assertRaises(AssertionError):
            env = gym.make(
                "RoomEnv-v1",
                policies={
                    "memory_management": "RL",
                    "question_answer": "RL",
                    "encoding": "argmax",
                },
            )
            del env

    def test_wrong_init1(self) -> None:

        with self.assertRaises(NotImplementedError):
            env = gym.make(
                "RoomEnv-v1",
                policies={
                    "memory_management": "episodic",
                    "question_answer": "episodic",
                    "encoding": "rl",
                },
            )
            del env

    def test_wrong_init2(self) -> None:

        with self.assertRaises(NotImplementedError):
            env = gym.make("RoomEnv-v1", observation_params="foo")
            state, info = env.reset()
            del env

    def test_rewards(self) -> None:
        env = gym.make(
            "RoomEnv-v1",
            policies={
                "memory_management": "RL",
                "question_answer": "episodic_semantic",
                "encoding": "argmax",
            },
            seed=random.randint(0, 10000),
            question_prob=0.1,
            capacity={"episodic": 16, "semantic": 16, "short": 1},
            varying_rewards=True,
        )
        state, info = env.reset()
        self.assertAlmostEqual(
            env.total_episode_rewards, env.CORRECT * env.num_questions
        )

    def test_reset_qa(self) -> None:
        for memory_management in ["episodic", "semantic"]:
            env = gym.make(
                "RoomEnv-v1",
                policies={
                    "memory_management": memory_management,
                    "question_answer": "rl",
                    "encoding": "argmax",
                },
                seed=random.randint(0, 10000),
                question_prob=0.1,
                capacity={"episodic": 16, "semantic": 16, "short": 1},
            )
            state, info = env.reset()
            self.assertIn("memory_systems", state)
            self.assertIn("question", state)

            self.assertIn("episodic", state["memory_systems"])
            self.assertIn("semantic", state["memory_systems"])
            self.assertIn("short", state["memory_systems"])
            self.assertIn("question", state)

            if memory_management == "semantic":
                self.assertGreater(len(state["memory_systems"]["semantic"]), 0)
                self.assertEqual(len(state["memory_systems"]["episodic"]), 0)

            else:
                self.assertEqual(len(state["memory_systems"]["semantic"]), 0)
                self.assertGreater(len(state["memory_systems"]["episodic"]), 0)

            self.assertEqual(len(state["memory_systems"]["short"]), 0)

            while True:
                state, reward, done, truncated, info = env.step(random.randint(0, 1))
                if done:
                    break
                self.assertIn("episodic", state["memory_systems"])
                self.assertIn("semantic", state["memory_systems"])
                self.assertIn("short", state["memory_systems"])
                self.assertIn("question", state)

    def test_reset_memory_management(self) -> None:
        for qa_policy in ["episodic", "semantic", "episodic_semantic"]:
            env = gym.make(
                "RoomEnv-v1",
                policies={
                    "memory_management": "rl",
                    "question_answer": qa_policy,
                    "encoding": "argmax",
                },
                seed=random.randint(0, 10000),
                question_prob=0.1,
                capacity={"episodic": 16, "semantic": 16, "short": 1},
            )
            state, info = env.reset()
            self.assertIn("episodic", state)
            self.assertIn("semantic", state)
            self.assertIn("short", state)

            self.assertEqual(len(state["semantic"]), 0)
            self.assertEqual(len(state["episodic"]), 0)
            self.assertEqual(len(state["short"]), 1)
            while True:
                state, reward, done, truncated, info = env.step(random.randint(0, 2))
                if done:
                    break
                self.assertIn("episodic", state)
                self.assertIn("semantic", state)
                self.assertIn("short", state)
