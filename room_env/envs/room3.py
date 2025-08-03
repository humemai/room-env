"""Room Environment v3 - Simplified deterministic version."""

import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
from rdflib import Graph, URIRef

from ..utils import is_running_notebook, rdf_to_list
from ..utils import read_json_prod as read_json


class RoomEnv3(gym.Env):
    """Simplified Room Environment v3 with deterministic object movement."""

    def __init__(
        self,
        terminates_at: int = 99,
        room_size: str = "small",
    ) -> None:
        """Initialize Room Environment v3.

        Args:
            terminates_at: When episode terminates (can be any value)
            room_size: Room configuration size
        """
        super().__init__()

        self.is_notebook = is_running_notebook()

        # Load configuration
        config = read_json(f"data/room-config-{room_size}-v3.json")

        self.grid_length = config["grid_length"]
        self.room_names = config["room_names"]
        self.room_positions = config["room_positions"]
        self.base_room_connections = config["room_connections"]
        self.static_names = config["static_names"]
        self.moving_names = config["moving_names"]
        self.initial_static_locations = config["static_locations"]
        self.initial_moving_locations = config["moving_locations"]
        self.movement_preferences = config["movement_preferences"]
        self.initial_agent_location = config["agent_location"]
        self.selected_walls = config["selected_walls"]

        # Convert wall configs from string keys back to tuples
        self.wall_configs = {}
        for wall_key, pattern in config["wall_configs"].items():
            # Convert string key back to tuple
            parts = wall_key.split("|")
            wall_tuple = (parts[0], parts[1], parts[2])
            self.wall_configs[wall_tuple] = pattern

        self.question_objects = config["question_objects"]  # Always 100 questions

        # Environment parameters
        self.terminates_at = terminates_at

        # Dummy gym spaces
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

        self.entities = (
            self.room_names + self.static_names + self.moving_names + ["agent", "wall"]
        )
        self.relations = ["north", "east", "south", "west", "at_location"]
        self.total_maximum_episode_rewards = self.terminates_at + 1

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0

        # Reset object locations
        self.static_locations = self.initial_static_locations.copy()
        self.moving_locations = self.initial_moving_locations.copy()
        self.agent_location = self.initial_agent_location

        # Set initial wall layout
        self._update_wall_layout()

        # Get initial observations
        observations = self._get_observations()
        info = {}

        return observations, info

    def step(self, actions):
        """Take environment step.

        Args:
            actions: Tuple of (question_answer, movement_action)

        Returns:
            observations, reward, done, truncated, info
        """
        question_answer, movement_action = actions

        # Calculate reward for current question
        reward = self._calculate_reward(question_answer)

        # Move objects deterministically
        self._move_objects()

        # Move agent
        self._move_agent(movement_action)

        # Update step counter
        self.current_step += 1

        # Update wall layout based on patterns
        self._update_wall_layout()

        # Check if done
        done = self.current_step > self.terminates_at
        truncated = False

        # Get next observations
        observations = self._get_observations()
        info = {}

        return observations, reward, done, truncated, info

    def _update_wall_layout(self):
        """Update room connections based on periodic wall patterns."""
        # Start with base connections
        self.room_connections = {}
        for room_name in self.room_names:
            self.room_connections[room_name] = self.base_room_connections[
                room_name
            ].copy()

        # Apply inner walls based on their patterns
        for wall, pattern in self.wall_configs.items():
            # Get pattern state at current step using actual pattern length
            pattern_length = len(pattern)
            pattern_index = self.current_step % pattern_length
            is_wall_active = pattern[pattern_index] == 1

            if is_wall_active:
                room1, room2, wall_type = wall
                if wall_type == "horizontal":
                    self.room_connections[room1]["south"] = "wall"
                    self.room_connections[room2]["north"] = "wall"
                else:  # vertical
                    self.room_connections[room1]["east"] = "wall"
                    self.room_connections[room2]["west"] = "wall"

    def _move_objects(self):
        """Move all moving objects deterministically."""
        for obj_name in self.moving_names:
            current_location = self.moving_locations[obj_name]
            preferences = self.movement_preferences[obj_name]

            # Try each preferred direction in order
            for direction in preferences:
                next_location = self.room_connections[current_location][direction]
                if next_location != "wall":
                    self.moving_locations[obj_name] = next_location
                    break

    def _move_agent(self, action):
        """Move agent based on action."""
        if action in ["north", "east", "south", "west"]:
            next_location = self.room_connections[self.agent_location][action]
            if next_location != "wall":
                self.agent_location = next_location
        # "stay" action or invalid action keeps agent in place

    def _get_observations(self):
        """Get current observations."""
        # Room layout from agent's perspective
        agent_room_obs = []
        for direction in ["north", "east", "south", "west"]:
            connected = self.room_connections[self.agent_location][direction]
            agent_room_obs.append([self.agent_location, direction, connected])

        # Objects in agent's room
        for obj_name in self.static_names + self.moving_names:
            obj_location = self.static_locations.get(
                obj_name
            ) or self.moving_locations.get(obj_name)
            if obj_location == self.agent_location:
                agent_room_obs.append([obj_name, "at_location", obj_location])

        # Agent location
        agent_room_obs.append(["agent", "at_location", self.agent_location])

        random.shuffle(agent_room_obs)

        # Generate exactly one question for this step (not a list)
        question_index = self.current_step % 100  # Cycle through 100 questions
        obj_name = self.question_objects[question_index]
        question = [obj_name, "at_location", "?"]

        return {"room": agent_room_obs, "question": question}

    def _calculate_reward(self, answer):
        """Calculate reward for question answer."""
        question_index = self.current_step % 100
        obj_name = self.question_objects[question_index]

        # Get correct location
        if obj_name in self.static_names:
            correct_location = self.static_locations[obj_name]
        else:
            correct_location = self.moving_locations[obj_name]

        # Check answer and return single reward
        return 1 if answer == correct_location else 0

    def _find_objects_in_room(self, room_name: str) -> list:
        """Find all objects currently in a specified room, categorized by type."""
        objects_in_room = []

        # Check static objects
        for obj_name in self.static_names:
            if self.static_locations[obj_name] == room_name:
                objects_in_room.append(("static", obj_name))

        # Check moving objects
        for obj_name in self.moving_names:
            if self.moving_locations[obj_name] == room_name:
                objects_in_room.append(("moving", obj_name))

        # Check agent
        if self.agent_location == room_name:
            objects_in_room.append(("agent", "agent"))

        return objects_in_room

    def render(
        self,
        render_mode: str = "grid",
        figsize: tuple[int, int] = (12, 12),
        cell_text_size: int = 12,
        save_fig_dir: str = "./DEBUG/",
        image_format: str = "png",
        graph_layout: str = "spring",
    ) -> None:
        """Render the current state of the environment.

        Args:
            render_mode: How to render ('console', 'grid', or 'graph')
            figsize: Size of the figure
            cell_text_size: Text size in cells
            save_fig_dir: Directory to save figures
            image_format: Format to save images in
            graph_layout: Layout for graph rendering ('spring', 'circular', 'kamada_kawai')
        """
        if render_mode == "console":
            print(f"Step {self.current_step}:")
            print(f"Agent location: {self.agent_location}")
            print(f"Static objects: {self.static_locations}")
            print(f"Moving objects: {self.moving_locations}")

        elif render_mode == "graph":
            self._render_graph(figsize, save_fig_dir, image_format, graph_layout)

        elif render_mode == "grid":
            plt.figure(figsize=figsize)

            # Define colors for different object types
            colors = {
                "room": "#FFE4B5",  # Moccasin (Soft Gold)
                "moving": "#90EE90",  # Light Green
                "static": "#87CEFA",  # Light Blue
                "agent": "#D8BFD8",  # Thistle (Soft Purple)
            }

            # Draw the grid
            for i in range(self.grid_length):
                for j in range(self.grid_length):
                    room_idx = i * self.grid_length + j
                    room_name = self.room_names[room_idx]

                    # Draw room cell with room color background
                    rect = plt.Rectangle(
                        (j, self.grid_length - 1 - i),
                        1,
                        1,
                        facecolor=colors["room"],
                        edgecolor="black",
                        linewidth=1,
                    )
                    plt.gca().add_patch(rect)

                    # Get objects in this room
                    objects_in_room = self._find_objects_in_room(room_name)

                    # Group objects by type for display
                    static_objs = [
                        obj for obj_type, obj in objects_in_room if obj_type == "static"
                    ]
                    moving_objs = [
                        obj for obj_type, obj in objects_in_room if obj_type == "moving"
                    ]
                    agent_objs = [
                        obj for obj_type, obj in objects_in_room if obj_type == "agent"
                    ]

                    y_offset = 0.85  # Start position for room name
                    line_height = 0.12  # Space between lines

                    # Display room name at top in black (no background box)
                    plt.text(
                        j + 0.5,
                        self.grid_length - 1 - i + y_offset,
                        room_name,
                        ha="center",
                        va="center",
                        fontsize=cell_text_size,
                        color="black",
                        weight="bold",
                    )
                    y_offset -= line_height * 1.5

                    # Display each static object separately with blue background
                    for obj in static_objs:
                        if y_offset > 0.1:  # Only display if there's space
                            plt.text(
                                j + 0.5,
                                self.grid_length - 1 - i + y_offset,
                                obj,
                                ha="center",
                                va="center",
                                fontsize=cell_text_size - 2,
                                color="black",
                                weight="bold",
                                bbox=dict(
                                    boxstyle="round,pad=0.15",
                                    facecolor=colors["static"],
                                    alpha=0.8,
                                ),
                            )
                            y_offset -= line_height

                    # Display each moving object separately with green background
                    for obj in moving_objs:
                        if y_offset > 0.1:  # Only display if there's space
                            plt.text(
                                j + 0.5,
                                self.grid_length - 1 - i + y_offset,
                                obj,
                                ha="center",
                                va="center",
                                fontsize=cell_text_size - 2,
                                color="black",
                                weight="bold",
                                bbox=dict(
                                    boxstyle="round,pad=0.15",
                                    facecolor=colors["moving"],
                                    alpha=0.8,
                                ),
                            )
                            y_offset -= line_height

                    # Display agent with purple background
                    for obj in agent_objs:
                        if y_offset > 0.1:  # Only display if there's space
                            plt.text(
                                j + 0.5,
                                self.grid_length - 1 - i + y_offset,
                                obj,
                                ha="center",
                                va="center",
                                fontsize=cell_text_size - 1,
                                color="black",
                                weight="bold",
                                bbox=dict(
                                    boxstyle="round,pad=0.15",
                                    facecolor=colors["agent"],
                                    alpha=0.8,
                                ),
                            )

            # Draw walls with thicker lines
            self._draw_walls()

            plt.gca().set_xlim(0, self.grid_length)
            plt.gca().set_ylim(0, self.grid_length)
            plt.gca().set_aspect("equal")

            # Remove axis ticks and labels
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])

            plt.grid(True, alpha=0.3)

            plt.title(f"Bird Eye View - Step {self.current_step}")

            # Save figure
            if save_fig_dir is not None:
                os.makedirs(save_fig_dir, exist_ok=True)
                filename = f"bird-eye-view_step_{str(self.current_step).zfill(3)}.{image_format}"
                plt.savefig(
                    os.path.join(save_fig_dir, filename), dpi=150, bbox_inches="tight"
                )

            plt.show()

    def separate_overlapping_nodes(self, pos, min_distance=0.1, max_iterations=50):
        """Separate overlapping nodes by applying small adjustments to their positions."""
        import random

        import numpy as np

        nodes = list(pos.keys())
        adjusted_pos = pos.copy()

        for iteration in range(max_iterations):
            overlaps_found = False

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node1, node2 = nodes[i], nodes[j]
                    pos1 = np.array(adjusted_pos[node1])
                    pos2 = np.array(adjusted_pos[node2])

                    distance = np.linalg.norm(pos1 - pos2)

                    if distance < min_distance:
                        overlaps_found = True

                        # Calculate separation vector
                        if distance == 0:
                            # If nodes are exactly on top of each other, use random direction
                            angle = random.uniform(0, 2 * np.pi)
                            separation = (
                                np.array([np.cos(angle), np.sin(angle)]) * min_distance
                            )
                        else:
                            # Move nodes apart along the line connecting them
                            direction = (pos1 - pos2) / distance
                            separation = direction * (min_distance - distance) / 2

                        # Apply separation
                        adjusted_pos[node1] = pos1 + separation
                        adjusted_pos[node2] = pos2 - separation

            if not overlaps_found:
                break

        return adjusted_pos

    def _render_graph(
        self,
        figsize: tuple[int, int] = (12, 12),
        save_fig_dir: str = "./DEBUG/",
        image_format: str = "png",
        layout: str = "kamada_kawai",
    ) -> None:
        """Render the RDF graph as a network visualization."""
        # Get RDF graph and convert to networkx
        rdf_graph = self.get_rdf_graph()
        triples = rdf_to_list(rdf_graph)

        # Create networkx graph
        G = nx.DiGraph()

        # Add nodes and edges
        for subject, predicate, obj in triples:
            G.add_edge(subject, obj, label=predicate)

        # Create figure
        plt.figure(figsize=figsize)

        # Define node colors based on type
        node_colors = []
        color_mapping = {
            "agent": "#D8BFD8",  # Thistle (Soft Purple)
            "wall": "#D3D3D3",  # Light Gray
        }

        # Color rooms
        room_color = "#FFE4B5"  # Moccasin (Soft Gold)
        static_color = "#87CEFA"  # Light Blue
        moving_color = "#90EE90"  # Light Green

        for node in G.nodes():
            if node in self.room_names:
                node_colors.append(room_color)
            elif node in self.static_names:
                node_colors.append(static_color)
            elif node in self.moving_names:
                node_colors.append(moving_color)
            elif node == "agent":
                node_colors.append(color_mapping["agent"])
            elif node == "wall":
                node_colors.append(color_mapping["wall"])
            else:
                node_colors.append("#FFFFFF")  # White for unknown

        # Choose layout with better spacing
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            raise ValueError(f"Unknown layout: {layout}.")

        # Separate overlapping nodes
        pos = self.separate_overlapping_nodes(
            pos, min_distance=0.15, max_iterations=100
        )

        # Draw nodes with larger size for better visibility
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
            edgecolors="black",
            linewidths=2,
        )

        # Draw node labels with consistent styling
        nx.draw_networkx_labels(
            G, pos, font_size=12, font_weight="bold", font_color="black"
        )

        # Draw edges with better spacing
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color="gray",
            arrows=True,
            arrowsize=25,
            arrowstyle="->",
            alpha=0.7,
            width=2,
            connectionstyle="arc3,rad=0.1",  # Slight curve to avoid overlap
        )

        # Draw edge labels with consistent black text and same font size
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels,
            font_size=12,
            font_color="black",
            font_weight="bold",
            alpha=1.0,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

        plt.title(
            f"Knowledge Graph View - Step {self.current_step}",
            fontsize=16,
            fontweight="bold",
        )
        plt.axis("off")
        plt.tight_layout()

        # Save figure
        if save_fig_dir is not None:
            os.makedirs(save_fig_dir, exist_ok=True)
            filename = (
                f"graph-view_step_{str(self.current_step).zfill(3)}.{image_format}"
            )
            plt.savefig(
                os.path.join(save_fig_dir, filename), dpi=150, bbox_inches="tight"
            )

        plt.show()

    def _draw_walls(self):
        """Draw outer walls and inner walls with thick black lines."""
        # Outer walls - very thick black
        wall_thickness = 8

        # Top wall
        plt.plot(
            [0, self.grid_length],
            [self.grid_length, self.grid_length],
            "k-",
            linewidth=wall_thickness,
        )
        # Bottom wall
        plt.plot([0, self.grid_length], [0, 0], "k-", linewidth=wall_thickness)
        # Left wall
        plt.plot([0, 0], [0, self.grid_length], "k-", linewidth=wall_thickness)
        # Right wall
        plt.plot(
            [self.grid_length, self.grid_length],
            [0, self.grid_length],
            "k-",
            linewidth=wall_thickness,
        )

        inner_wall_thickness = 4

        # Draw active inner walls based on current patterns
        for wall, pattern in self.wall_configs.items():
            pattern_length = len(pattern)
            pattern_index = self.current_step % pattern_length
            is_wall_active = pattern[pattern_index] == 1

            if is_wall_active:
                room1, room2, wall_type = wall
                pos1 = self.room_positions[room1]
                pos2 = self.room_positions[room2]

                if wall_type == "horizontal":
                    # Wall between vertically adjacent rooms
                    i1, j1 = pos1
                    i2, j2 = pos2
                    y = self.grid_length - max(i1, i2)  # Convert to plot coordinates
                    x_start = j1
                    x_end = j1 + 1
                    plt.plot(
                        [x_start, x_end], [y, y], "k-", linewidth=inner_wall_thickness
                    )

                else:  # vertical
                    # Wall between horizontally adjacent rooms
                    i1, j1 = pos1
                    i2, j2 = pos2
                    x = max(j1, j2)  # Convert to plot coordinates
                    y_start = self.grid_length - i1 - 1
                    y_end = self.grid_length - i1
                    plt.plot(
                        [x, x], [y_start, y_end], "k-", linewidth=inner_wall_thickness
                    )

    def get_rdf_graph(self):
        """Get current state as RDF graph."""
        g = Graph()

        # Add room connections
        for room_name, connections in self.room_connections.items():
            for direction, connected in connections.items():
                g.add((URIRef(room_name), URIRef(direction), URIRef(connected)))
        # Add object locations
        for obj_name, location in self.static_locations.items():
            g.add((URIRef(obj_name), URIRef("at_location"), URIRef(location)))

        for obj_name, location in self.moving_locations.items():
            g.add((URIRef(obj_name), URIRef("at_location"), URIRef(location)))

        # Add agent location
        g.add((URIRef("agent"), URIRef("at_location"), URIRef(self.agent_location)))

        return g
