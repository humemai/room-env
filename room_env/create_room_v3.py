"""Room Creator for v3 - Simplified square grid with deterministic behavior."""

import random
from itertools import combinations

from .utils import read_lines
from .utils import write_json_prod as write_json


class RoomCreator:
    def __init__(
        self,
        filename: str = "dev",
        grid_length: int = 3,
        num_static_objects: int = 3,
        num_moving_objects: int = 3,
        num_inner_walls: int = 3,
        seed: int = 42,
    ) -> None:
        """Create a simplified square room environment.

        Args:
            filename: Config filename to save
            grid_length: Size of square grid (must be odd)
            num_static_objects: Number of static objects
            num_moving_objects: Number of moving objects
            num_inner_walls: Number of inner walls to select
            seed: Random seed for reproducibility
        """
        assert grid_length % 2 == 1, "grid_length must be odd"

        self.filename = filename
        self.grid_length = grid_length
        self.num_rooms = grid_length**2
        self.num_static_objects = num_static_objects
        self.num_moving_objects = num_moving_objects
        self.num_inner_walls = num_inner_walls
        self.seed = seed

        # Define the 10 possible wall patterns
        self.wall_patterns = [
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
            [1, 0],
            [1, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 0],
            [1, 1, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 0, 0, 1],
        ]

        random.seed(self.seed)

    def run(self):
        """Create and save the room configuration."""
        self._create_room_grid()
        self._create_room_connections()
        self._create_object_configs()
        self._create_wall_configs()
        self._create_question_list()
        self._save_config()

    def _create_room_grid(self):
        """Create square grid of rooms."""
        # Load names
        room_names = read_lines("data/room_names.txt")[: self.num_rooms]
        static_names = read_lines("data/static_objects.txt")[: self.num_static_objects]
        moving_names = read_lines("data/moving_objects.txt")[: self.num_moving_objects]

        self.room_names = room_names
        self.static_names = static_names
        self.moving_names = moving_names

        # Create grid positions
        self.room_positions = {}
        for i in range(self.grid_length):
            for j in range(self.grid_length):
                room_idx = i * self.grid_length + j
                self.room_positions[room_names[room_idx]] = (i, j)

    def _create_room_connections(self):
        """Create room connections with outer walls."""
        self.room_connections = {}

        for room_name, (i, j) in self.room_positions.items():
            connections = {
                "north": "wall",
                "east": "wall",
                "south": "wall",
                "west": "wall",
            }

            # North
            if i > 0:
                north_idx = (i - 1) * self.grid_length + j
                connections["north"] = self.room_names[north_idx]

            # East
            if j < self.grid_length - 1:
                east_idx = i * self.grid_length + (j + 1)
                connections["east"] = self.room_names[east_idx]

            # South
            if i < self.grid_length - 1:
                south_idx = (i + 1) * self.grid_length + j
                connections["south"] = self.room_names[south_idx]

            # West
            if j > 0:
                west_idx = i * self.grid_length + (j - 1)
                connections["west"] = self.room_names[west_idx]

            self.room_connections[room_name] = connections

    def _create_object_configs(self):
        """Create object initial locations and movement preferences."""
        # First, determine the center room where agent will start
        center_idx = self.grid_length // 2
        center_room = self.room_names[center_idx * self.grid_length + center_idx]
        self.agent_location = center_room

        # Available rooms exclude the center room (where agent starts)
        available_rooms = [room for room in self.room_names if room != center_room]

        # Static objects - ensure each is in a different room
        self.static_locations = {}
        for obj_name in self.static_names:
            room = random.choice(available_rooms)
            self.static_locations[obj_name] = room
            available_rooms.remove(room)

        # Moving objects - ensure each is in a different room from static objects and each other
        self.moving_locations = {}
        self.movement_preferences = {}
        directions = ["north", "east", "south", "west"]

        for obj_name in self.moving_names:
            room = random.choice(available_rooms)
            self.moving_locations[obj_name] = room
            available_rooms.remove(room)

            preferences = directions.copy()
            random.shuffle(preferences)
            self.movement_preferences[obj_name] = preferences

    def _set_agent_location(self):
        """Set agent starting location to center room."""
        # Calculate center position
        center_idx = self.grid_length // 2
        center_room = self.room_names[center_idx * self.grid_length + center_idx]
        self.agent_location = center_room

    def _create_wall_configs(self):
        """Create periodic inner wall configurations."""
        # Get all possible inner wall positions
        possible_walls = []

        # Horizontal walls (between vertically adjacent rooms)
        for i in range(self.grid_length - 1):
            for j in range(self.grid_length):
                room1 = i * self.grid_length + j
                room2 = (i + 1) * self.grid_length + j
                possible_walls.append(
                    (self.room_names[room1], self.room_names[room2], "horizontal")
                )

        # Vertical walls (between horizontally adjacent rooms)
        for i in range(self.grid_length):
            for j in range(self.grid_length - 1):
                room1 = i * self.grid_length + j
                room2 = i * self.grid_length + (j + 1)
                possible_walls.append(
                    (self.room_names[room1], self.room_names[room2], "vertical")
                )

        # Ensure we don't select more walls than possible
        max_walls = len(possible_walls)
        if self.num_inner_walls > max_walls:
            self.num_inner_walls = max_walls
            print(f"Warning: Reduced num_inner_walls to {max_walls} (maximum possible)")

        # Select walls that maintain connectivity
        self.selected_walls = []
        attempts = 0
        max_attempts = 1000

        while (
            len(self.selected_walls) < self.num_inner_walls and attempts < max_attempts
        ):
            # Randomly select remaining walls
            remaining_walls = [
                w for w in possible_walls if w not in self.selected_walls
            ]
            candidate_wall = random.choice(remaining_walls)

            # Test if adding this wall maintains connectivity
            test_walls = self.selected_walls + [candidate_wall]
            if self._test_connectivity_with_all_walls_on(test_walls):
                self.selected_walls.append(candidate_wall)

            attempts += 1

        if len(self.selected_walls) < self.num_inner_walls:
            print(
                f"Warning: Could only select {len(self.selected_walls)} walls that maintain connectivity"
            )

        # Assign patterns to selected walls - convert tuples to strings for JSON serialization
        self.wall_configs = {}
        for wall in self.selected_walls:
            pattern = random.choice(self.wall_patterns)
            # Convert tuple to string key for JSON compatibility
            wall_key = f"{wall[0]}|{wall[1]}|{wall[2]}"
            self.wall_configs[wall_key] = pattern

    def _test_connectivity_with_all_walls_on(self, walls):
        """Test if rooms remain connected when all specified walls are active."""
        # Create temporary room connections with all walls on
        temp_connections = {}
        for room_name in self.room_names:
            temp_connections[room_name] = self.room_connections[room_name].copy()

        # Apply all walls
        for room1, room2, wall_type in walls:
            if wall_type == "horizontal":
                temp_connections[room1]["south"] = "wall"
                temp_connections[room2]["north"] = "wall"
            else:  # vertical
                temp_connections[room1]["east"] = "wall"
                temp_connections[room2]["west"] = "wall"

        # Check connectivity using DFS
        visited = set()
        start_room = self.room_names[0]
        self._dfs(start_room, temp_connections, visited)

        return len(visited) == len(self.room_names)

    def _dfs(self, room, connections, visited):
        """Depth-first search for connectivity check."""
        visited.add(room)
        for direction in ["north", "east", "south", "west"]:
            neighbor = connections[room][direction]
            if neighbor != "wall" and neighbor not in visited:
                self._dfs(neighbor, connections, visited)

    def _create_question_list(self):
        """Create list of exactly 100 random questions."""
        all_objects = self.static_names + self.moving_names
        self.question_objects = [random.choice(all_objects) for _ in range(100)]

    def _save_config(self):
        """Save configuration to JSON file."""
        config = {
            "grid_length": self.grid_length,
            "room_names": self.room_names,
            "room_positions": self.room_positions,
            "room_connections": self.room_connections,
            "static_names": self.static_names,
            "moving_names": self.moving_names,
            "static_locations": self.static_locations,
            "moving_locations": self.moving_locations,
            "movement_preferences": self.movement_preferences,
            "agent_location": self.agent_location,
            "selected_walls": self.selected_walls,
            "wall_configs": self.wall_configs,
            "question_objects": self.question_objects,
            "seed": self.seed,
        }

        write_json(config, f"data/room-config-{self.filename}-v3.json")
        print(f"Saved configuration to room-config-{self.filename}-v3.json")
        print(f"Selected {len(self.selected_walls)} inner walls with periodic patterns")
