import os
import sys
import unittest

# Add the parent directory to the path so we can import room_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from room_env.create_room_v3 import RoomCreator
from room_env.envs.room3 import RoomEnv3


class TestRoomEnv3(unittest.TestCase):
    """Test suite for Room Environment v3."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a dev environment for testing
        room_creator = RoomCreator(
            filename="test",
            grid_length=3,
            num_static_objects=2,
            num_moving_objects=2,
            num_inner_walls=4,
            seed=42,  # Fixed seed for reproducible tests
        )
        room_creator.run()
        
        # Create environment
        self.env_config = {
            "terminates_at": 25,
            "room_size": "test",
        }
        self.env = RoomEnv3(**self.env_config)

    def test_environment_initialization(self):
        """Test that environment initializes correctly."""
        observations, info = self.env.reset()
        
        # Check basic properties
        self.assertEqual(self.env.grid_length, 3)
        self.assertEqual(len(self.env.room_names), 9)  # 3x3 grid
        self.assertEqual(len(self.env.static_names), 2)
        self.assertEqual(len(self.env.moving_names), 2)
        self.assertEqual(self.env.current_step, 0)
        
        # Check observations structure
        self.assertIn("room", observations)
        self.assertIn("question", observations)
        
        # Check agent starts in center room
        center_room = self.env.room_names[4]  # Middle of 3x3 grid
        self.assertEqual(self.env.agent_location, center_room)

    def test_deterministic_object_movement(self):
        """Test that moving objects follow deterministic movement patterns."""
        observations, info = self.env.reset()
        
        # Record initial positions
        initial_moving_locations = self.env.moving_locations.copy()
        
        # Run several steps and check movement follows preferences
        for step in range(10):
            # Track positions before step
            before_positions = self.env.moving_locations.copy()
            
            # Take step
            observations, reward, done, truncated, info = self.env.step(("living", "stay"))
            
            # After the step, get the room connections that were used for movement
            # (walls are updated during the step, so we need to reconstruct the state)
            
            # Check that each moving object tried to move according to preferences
            for obj_name in self.env.moving_names:
                preferences = self.env.movement_preferences[obj_name]
                old_location = before_positions[obj_name]
                new_location = self.env.moving_locations[obj_name]
                
                # To validate movement, we need to check what connections were available
                # We'll step back to the previous step's wall state to check
                previous_step = self.env.current_step - 1
                
                # Reconstruct room connections at the time movement happened
                temp_connections = {}
                for room_name in self.env.room_names:
                    temp_connections[room_name] = self.env.base_room_connections[room_name].copy()
                
                # Apply walls that were active at previous step
                for wall, pattern in self.env.wall_configs.items():
                    pattern_index = previous_step % 10
                    is_wall_active = pattern[pattern_index] == 1
                    
                    if is_wall_active:
                        room1, room2, wall_type = wall
                        if wall_type == "horizontal":
                            temp_connections[room1]["south"] = "wall"
                            temp_connections[room2]["north"] = "wall"
                        else:  # vertical
                            temp_connections[room1]["east"] = "wall"
                            temp_connections[room2]["west"] = "wall"
                
                # Find the first valid direction in preferences using correct connections
                expected_location = old_location  # Default: stay in place
                for direction in preferences:
                    next_location = temp_connections[old_location][direction]
                    if next_location != "wall":
                        expected_location = next_location
                        break  # Take the first valid direction
                
                # Object should be at the expected location
                self.assertEqual(new_location, expected_location)

    def test_periodic_wall_behavior(self):
        """Test that inner walls follow 10-step periodic patterns."""
        observations, info = self.env.reset()
        
        # Record wall patterns at step 0
        initial_connections = {}
        for room_name in self.env.room_names:
            initial_connections[room_name] = self.env.room_connections[room_name].copy()
        
        # Test that patterns repeat every 10 steps
        connections_at_steps = {}
        
        for step in range(20):  # Test 2 full cycles
            step_connections = {}
            for room_name in self.env.room_names:
                step_connections[room_name] = self.env.room_connections[room_name].copy()
            connections_at_steps[step] = step_connections
            
            # Take step
            observations, reward, done, truncated, info = self.env.step(("living", "stay"))
        
        # Check that step 0 and step 10 have same wall layout
        self.assertEqual(connections_at_steps[0], connections_at_steps[10])
        
        # Check that step 5 and step 15 have same wall layout
        self.assertEqual(connections_at_steps[5], connections_at_steps[15])

    def test_question_cycling_and_rewards(self):
        """Test that questions cycle through 100 questions and rewards are calculated correctly."""
        observations, info = self.env.reset()
        
        # Test that we get exactly 100 unique question indices before cycling
        question_objects = []
        rewards = []
        
        for step in range(105):  # Go beyond 100 to test cycling
            # Get current question
            question = observations["question"]
            question_obj = question[0]
            question_objects.append(question_obj)
            
            # Get correct answer for this question
            if question_obj in self.env.static_names:
                correct_answer = self.env.static_locations[question_obj]
            else:
                correct_answer = self.env.moving_locations[question_obj]
            
            # Test correct answer gives reward 1
            observations, reward, done, truncated, info = self.env.step((correct_answer, "stay"))
            rewards.append(reward)
            self.assertEqual(reward, 1)
            
            if done:
                break
        
        # Check that questions cycle properly
        self.assertEqual(len(self.env.question_objects), 100)
        
        # Questions at step 0 and 100 should be the same
        if len(question_objects) > 100:
            self.assertEqual(question_objects[0], question_objects[100])

    def test_incorrect_answers_give_zero_reward(self):
        """Test that incorrect answers give zero reward."""
        observations, info = self.env.reset()
        
        for step in range(5):
            # Always give wrong answer
            wrong_answer = "nonexistent_room"
            observations, reward, done, truncated, info = self.env.step((wrong_answer, "stay"))
            self.assertEqual(reward, 0)

    def test_agent_movement_and_walls(self):
        """Test agent movement respects walls."""
        observations, info = self.env.reset()
        
        initial_location = self.env.agent_location
        
        # Test each direction
        for direction in ["north", "east", "south", "west"]:
            # Reset to known position
            self.env.agent_location = initial_location
            
            # Try to move in direction
            expected_location = self.env.room_connections[initial_location][direction]
            
            observations, reward, done, truncated, info = self.env.step(("living", direction))
            
            if expected_location == "wall":
                # Should stay in same location if hitting wall
                self.assertEqual(self.env.agent_location, initial_location)
            else:
                # Should move to connected room
                self.assertEqual(self.env.agent_location, expected_location)

    def test_stay_action(self):
        """Test that stay action keeps agent in same location."""
        observations, info = self.env.reset()
        
        initial_location = self.env.agent_location
        observations, reward, done, truncated, info = self.env.step(("living", "stay"))
        
        self.assertEqual(self.env.agent_location, initial_location)

    def test_episode_termination(self):
        """Test that episode terminates at specified step."""
        observations, info = self.env.reset()
        
        done = False
        step = 0
        
        while not done and step <= self.env.terminates_at + 5:  # Safety limit
            observations, reward, done, truncated, info = self.env.step(("living", "stay"))
            step += 1
        
        self.assertTrue(done)
        self.assertEqual(step, self.env.terminates_at + 1)

    def test_static_objects_never_move(self):
        """Test that static objects never change location."""
        observations, info = self.env.reset()
        
        initial_static_locations = self.env.static_locations.copy()
        
        # Run for many steps
        for step in range(20):
            observations, reward, done, truncated, info = self.env.step(("living", "stay"))
            
            # Static objects should never move
            self.assertEqual(self.env.static_locations, initial_static_locations)

    def test_observations_structure(self):
        """Test that observations have correct structure."""
        observations, info = self.env.reset()
        
        # Check room observations
        self.assertIn("room", observations)
        room_obs = observations["room"]
        self.assertIsInstance(room_obs, list)
        
        # Should contain agent location and room connections
        agent_location_found = False
        room_connections_found = 0
        
        for obs in room_obs:
            self.assertIsInstance(obs, list)
            self.assertEqual(len(obs), 3)
            
            if obs[0] == "agent" and obs[1] == "at_location":
                agent_location_found = True
            elif obs[1] in ["north", "east", "south", "west"]:
                room_connections_found += 1
        
        self.assertTrue(agent_location_found)
        self.assertEqual(room_connections_found, 4)  # 4 directions
        
        # Check question (single question, not a list)
        self.assertIn("question", observations)
        question = observations["question"]
        self.assertIsInstance(question, list)
        self.assertEqual(len(question), 3)
        self.assertEqual(question[1], "at_location")
        self.assertEqual(question[2], "?")

    def test_wall_configuration_loading(self):
        """Test that wall configurations are loaded correctly."""
        self.assertIsInstance(self.env.wall_configs, dict)
        self.assertTrue(len(self.env.wall_configs) > 0)
        
        # Each wall config should have 10 steps of 0s and 1s
        for wall, pattern in self.env.wall_configs.items():
            self.assertEqual(len(pattern), 10)
            self.assertTrue(all(x in [0, 1] for x in pattern))
            self.assertIsInstance(wall, tuple)
            self.assertEqual(len(wall), 3)  # (room1, room2, wall_type)
            