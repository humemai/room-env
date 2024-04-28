"""Only run this in Juputer notebook."""

import random
from copy import deepcopy
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from .utils import find_connected_nodes
from .utils import read_json_prod as read_json
from .utils import write_json_prod as write_json

EPSILON = 1e-3


class RoomCreator:
    def __init__(
        self,
        grid_length: int = 1,
        num_rooms: int = 1,
        num_static_objects: int = 1,
        num_independent_objects: int = 1,
        num_dependent_objects: int = 1,
        room_prob: float = 0.7,
        minimum_transition_stay_prob: float = 0.5,
        give_fake_names: bool = False,
        static_object_in_every_room: bool = True,
        filename: str = "dev",
    ) -> None:
        """Create rooms with objects.

        Args:
            grid_length: grid legnth
            num_rooms: number of rooms in the grid
            num_static_objects: Number of static objects to create.
            num_independent_objects: Number of independent objects to create.
            num_dependent_objects: Number of dependent objects to create.
            room_prob: probability of a cell being a room.
            minimum_transition_stay_prob: probability of staying in the same room.
                This should be quite high. Otherwise, the objects are not tractable.
            give_fake_names: If True, give fake names to the rooms and objects.
            static_object_in_every_room: If True, every room has at least one static
                object.
            filename: filename to save the config.

        """
        self.filename = filename
        print(
            f"The config will be saved as "
            f"./room_env/data/room-config-{self.filename}-v2.json\n"
        )
        self.grid_length = grid_length
        self.num_rooms = num_rooms
        self.num_static_objects = num_static_objects
        self.num_independent_objects = num_independent_objects
        self.num_dependent_objects = num_dependent_objects
        self.room_prob = room_prob
        self.give_fake_names = give_fake_names
        self.minimum_transition_stay_prob = minimum_transition_stay_prob
        self.static_object_in_every_room = static_object_in_every_room

    def run(self):
        self._create_grid_world()
        self._create_room_config()
        self._create_object_init_config()
        self._create_object_transition_config()
        self._give_names()
        self._create_question_probs()
        write_json(
            {
                "room_config": self.room_config_str,
                "object_init_config": self.object_init_config_str,
                "object_transition_config": self.object_transition_config_str,
                "object_question_probs": self.question_probs,
                "grid": self.grid,
                "room_indexes": self.room_indexes,
                "names": self.names,
            },
            f"./data/room-config-{self.filename}-v2.json",
        )

        self._visualize_grids()

    def _create_grid_world(self) -> None:
        rows = self.grid_length
        cols = self.grid_length

        while True:
            self.grid = [
                [1 if random.random() < self.room_prob else 0 for _ in range(cols)]
                for _ in range(rows)
            ]
            if sum(sum(row) for row in self.grid) == 0:
                continue

            self.connected_components = find_connected_nodes(self.grid)

            self.previous_grid = deepcopy(self.grid)

            self.room_indexes = max(
                enumerate(self.connected_components), key=lambda x: len(x[1])
            )[1]
            self.room_indexes.sort(key=lambda x: x[1])
            self.room_indexes.sort(key=lambda x: x[0])
            self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
            for i in range(rows):
                for j in range(cols):
                    if (i, j) in self.room_indexes:
                        self.grid[i][j] = 1

            # This is to assure that every room as at least one static object.
            if len(self.room_indexes) == self.num_rooms:
                if self.static_object_in_every_room:
                    if self.num_static_objects >= self.num_rooms:
                        break
                else:
                    break

    def _create_room_config(self) -> None:
        """Create a room configuration."""
        self.room_config = {
            i: {"north": "wall", "east": "wall", "south": "wall", "west": "wall"}
            for i in range(self.num_rooms)
        }
        for i, (row_idx, col_idx) in enumerate(self.room_indexes):
            north_idx = (row_idx - 1, col_idx)
            east_idx = (row_idx, col_idx + 1)
            south_idx = (row_idx + 1, col_idx)
            west_idx = (row_idx, col_idx - 1)

            if north_idx in self.room_indexes:
                self.room_config[i]["north"] = self.room_indexes.index(north_idx)
            if east_idx in self.room_indexes:
                self.room_config[i]["east"] = self.room_indexes.index(east_idx)
            if south_idx in self.room_indexes:
                self.room_config[i]["south"] = self.room_indexes.index(south_idx)
            if west_idx in self.room_indexes:
                self.room_config[i]["west"] = self.room_indexes.index(west_idx)

    def _create_object_init_config(self) -> None:
        """Create an object initialization configuration."""
        self.object_init_config = {}

        # This is made to be dertministic.
        for object_type, num_objects in zip(
            ["static", "independent", "dependent", "agent"],
            [
                self.num_static_objects,
                self.num_independent_objects,
                self.num_dependent_objects,
                1,
            ],
        ):
            self.object_init_config[object_type] = {
                object_num: {room_num: 0 for room_num in range(self.num_rooms)}
                for object_num in range(num_objects)
            }

        for object_num in range(self.num_static_objects):
            if self.static_object_in_every_room and object_num < self.num_rooms:
                room_num = object_num
                self.object_init_config["static"][object_num][room_num] = 1
            else:
                room_num = random.randint(0, self.num_rooms - 1)
                self.object_init_config["static"][object_num][room_num] = 1

        for object_type, num_objects in zip(
            ["independent", "dependent"],
            [
                self.num_independent_objects,
                self.num_dependent_objects,
            ],
        ):
            for object_num in range(num_objects):
                room_num = random.randint(0, self.num_rooms - 1)
                self.object_init_config[object_type][object_num][room_num] = 1

        self.object_init_config["agent"][0][0] = 1

        for object_type in ["static", "independent", "dependent", "agent"]:
            for _, room_num_dist in self.object_init_config[object_type].items():
                assert (
                    abs(sum(room_num_dist.values()) - 1) < EPSILON
                ), f"{sum(room_num_dist.values())}"

    def _create_object_transition_config(self) -> None:
        """Create an object transition configuration."""
        self.object_transition_config = {}
        self.object_transition_config["static"] = {
            i: None for i in range(self.num_static_objects)
        }
        self.object_transition_config["independent"] = {
            object_num: {
                room_num: {
                    "north": 0,
                    "east": 0,
                    "south": 0,
                    "west": 0,
                    "west": 0,
                    "stay": 0,
                }
                for room_num in range(self.num_rooms)
            }
            for object_num in range(self.num_independent_objects)
        }
        for object_num, room_num_nesws in self.object_transition_config[
            "independent"
        ].items():
            for room_num, nesw in room_num_nesws.items():
                nesw["stay"] = random.uniform(self.minimum_transition_stay_prob, 1)

                for direction, _ in nesw.items():
                    if (
                        direction != "stay"
                        and self.room_config[room_num][direction] != "wall"
                    ):
                        nesw[direction] = np.random.beta(0.5, 0.5, 1).item()

                denominator = sum(
                    [num for direction, num in nesw.items() if direction != "stay"]
                )

                denominator /= 1 - nesw["stay"]

                if denominator != 0:
                    for direction, _ in nesw.items():
                        if direction != "stay":
                            nesw[direction] = nesw[direction] / denominator
                else:
                    nesw["stay"] = 1

        self.object_transition_config["dependent"] = {
            dep_object_num: {
                ind_object_num: np.random.beta(0.5, 0.5, 1).item()
                for ind_object_num in range(self.num_independent_objects)
            }
            for dep_object_num in range(self.num_dependent_objects)
        }
        self.object_transition_config["agent"] = {"agent": None}

    def _give_names(self) -> None:
        """Give (fake) names to the rooms and objects."""

        self.names = {}
        if self.give_fake_names:
            fake_names = read_json("./data/names-v2.json")

            for object_type, num_objects in zip(
                ["room", "static_objects", "independent_objects", "dependent_objects"],
                [
                    self.num_rooms,
                    self.num_static_objects,
                    self.num_independent_objects,
                    self.num_dependent_objects,
                ],
            ):
                assert num_objects <= len(fake_names[object_type]), (
                    f"{num_objects} should be lower than or equal to "
                    f"{len(fake_names[object_type])}"
                )

                self.names[object_type] = random.sample(
                    fake_names[object_type], k=num_objects
                )
        else:
            for object_type, num_objects in zip(
                ["room", "static_objects", "independent_objects", "dependent_objects"],
                [
                    self.num_rooms,
                    self.num_static_objects,
                    self.num_independent_objects,
                    self.num_dependent_objects,
                ],
            ):
                if object_type == "room":
                    string = "room"
                elif object_type == "static_objects":
                    string = "sta"
                elif object_type == "independent_objects":
                    string = "ind"
                elif object_type == "dependent_objects":
                    string = "dep"
                self.names[object_type] = [
                    string + "_" + str(object_num).zfill(3)
                    for object_num in range(num_objects)
                ]

        print(
            f"# of rooms: {len(self.names['room'])}\n"
            f"# of static objects: {len(self.names['static_objects'])} \n"
            f"# of independent objects: {len(self.names['independent_objects'])} \n"
            f"# of dependent objects: {len(self.names['dependent_objects'])} \n"
        )
        pprint(self.names)

        self.room_config_str = {}
        for room_num in range(self.num_rooms):
            self.room_config_str[self.names["room"][room_num]] = self.room_config[
                room_num
            ]

            self.room_config_str[self.names["room"][room_num]] = {
                direction: (
                    self.names["room"][room_num_or_wall]
                    if room_num_or_wall != "wall"
                    else room_num_or_wall
                )
                for direction, room_num_or_wall in self.room_config_str[
                    self.names["room"][room_num]
                ].items()
            }

        self.object_init_config_str = {}
        self.object_init_config_str["static"] = {
            self.names["static_objects"][object_num]: {
                self.names["room"][room_num]: prob
                for room_num, prob in room_num_dist.items()
            }
            for object_num, room_num_dist in self.object_init_config["static"].items()
        }

        self.object_init_config_str["independent"] = {
            self.names["independent_objects"][object_num]: {
                self.names["room"][room_num]: prob
                for room_num, prob in nesws_dist.items()
            }
            for object_num, nesws_dist in self.object_init_config["independent"].items()
        }

        self.object_init_config_str["dependent"] = {
            self.names["dependent_objects"][object_num]: {
                self.names["room"][room_num]: prob
                for room_num, prob in room_num_dist.items()
            }
            for object_num, room_num_dist in self.object_init_config[
                "dependent"
            ].items()
        }

        self.object_init_config_str["agent"] = {
            "agent": {
                self.names["room"][room_num]: prob
                for room_num, prob in self.object_init_config["agent"][0].items()
            }
        }

        for object_type in ["static", "independent", "dependent", "agent"]:
            for _, room_num_dist in self.object_init_config_str[object_type].items():
                assert (
                    abs(sum(room_num_dist.values()) - 1) < EPSILON
                ), f"{sum(room_num_dist.values())} should be close to 1"

        self.object_transition_config_str = {}

        self.object_transition_config_str["static"] = {
            self.names["static_objects"][object_num]: None
            for object_num in self.object_transition_config["static"]
        }

        self.object_transition_config_str["independent"] = {
            self.names["independent_objects"][object_num]: {
                self.names["room"][room_num]: dist
                for room_num, dist in room_num_dist.items()
            }
            for object_num, room_num_dist in self.object_transition_config[
                "independent"
            ].items()
        }

        self.object_transition_config_str["dependent"] = {
            self.names["dependent_objects"][object_num]: {
                self.names["independent_objects"][human_num]: prob
                for human_num, prob in ind_dist.items()
            }
            for object_num, ind_dist in self.object_transition_config[
                "dependent"
            ].items()
        }
        self.object_transition_config_str["agent"] = self.object_transition_config[
            "agent"
        ]

    def _create_question_probs(self) -> None:
        """Create a question probability of an object being asked.

        This is applied in a global manner.

        """
        self.question_probs = {}

        for object_type in ["static", "independent", "dependent"]:
            self.question_probs[object_type] = {
                object_name: np.random.beta(0.5, 0.5, 1).item()
                for object_name in self.names[f"{object_type}_objects"]
            }

        # we don't want to ask a question about the agent.
        self.question_probs["agent"] = {"agent": 0}

        denominator = 0
        for object_type, object_probs in self.question_probs.items():
            denominator += sum(object_probs.values())

        if denominator == 0:
            raise ValueError("There should be at least one object to ask a question.")

        for object_type, object_probs in self.question_probs.items():
            for object_name, prob in object_probs.items():
                self.question_probs[object_type][object_name] = prob / denominator

        all_probs = [
            prob
            for object_probs in self.question_probs.values()
            for prob in object_probs.values()
        ]

        assert (
            abs(sum(all_probs) - 1) < EPSILON
        ), f"{sum(all_probs)} should be close to 1"

    def _visualize_grids(
        self, figsize: tuple[int, int] = (15, 15), cell_text_size: int = 10
    ) -> None:
        plt.figure(figsize=figsize)
        num_rows = len(self.grid)
        num_cols = len(self.grid[0])

        # Create the first subplot for the "Before" plot
        plt.subplot(121)
        plt.title("Before removing the disconnected rooms")

        for row in range(num_rows):
            for col in range(num_cols):
                cell_content = self.previous_grid[row][col]
                if cell_content != 0:
                    color = "white"
                else:
                    color = "black"
                plt.gca().add_patch(
                    plt.Rectangle((col, num_rows - 1 - row), 1, 1, facecolor=color)
                )

        plt.gca().set_aspect("equal")
        plt.gca().set_xticks(range(num_cols + 1))
        plt.gca().set_yticks(range(num_rows + 1))
        plt.gca().grid(which="both")

        # Create the second subplot for the "After" plot
        plt.subplot(122)
        plt.title("After removing the disconnected rooms")

        for row in range(num_rows):
            for col in range(num_cols):
                cell_content = self.grid[row][col]
                if cell_content != 0:
                    color = "white"
                    room_index = self.room_indexes.index((row, col))
                    text = "\n".join(self.names["room"][room_index].split("_"))

                else:
                    color = "black"
                    text = ""
                plt.gca().add_patch(
                    plt.Rectangle((col, num_rows - 1 - row), 1, 1, facecolor=color)
                )
                plt.text(
                    col + 0.5,
                    num_rows - 1 - row + 0.5,
                    text,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=cell_text_size,
                )
        plt.gca().set_aspect("equal")
        plt.gca().set_xticks(range(num_cols + 1))
        plt.gca().set_yticks(range(num_rows + 1))
        plt.gca().grid(which="both")

        plt.show()
