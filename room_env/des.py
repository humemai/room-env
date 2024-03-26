"""Discrete Event Simlator (RoomDes) for the Room."""

from copy import deepcopy
from pprint import pprint

from .utils import read_json_prod as read_json
from .utils import write_json


def fill_des_resources(des_size: str) -> None:
    """Fill resources

    Args:
        des_size

    """
    des = RoomDes(des_size=des_size, check_resources=False)
    des.run()
    resources = {
        foo: 9999
        for foo in set(
            [bar["object_location"] for foo in des.states for bar in foo.values()]
        )
    }
    des.config["resources"] = deepcopy(resources)
    write_json(des.config, f"./room_env/data/des-config-{des_size}.json")
    resources = []
    des = RoomDes(des_size=des_size, check_resources=True)
    resources.append(deepcopy(des.resources))
    while des.until > 0:
        des.step()
        des.until -= 1
        resources.append(deepcopy(des.resources))

    object_locations = deepcopy(list(des.resources.keys()))
    resources = {
        object_location: 9999
        - min([resource[object_location] for resource in resources])
        for object_location in object_locations
    }

    des.config["resources"] = deepcopy(resources)
    write_json(des.config, f"./room_env/data/des-config-{des_size}.json")
    des = RoomDes(des_size=des_size, check_resources=True)


class RoomDes:
    """RoomDes Class.

    This class is very simple at the moment. When it's initialized, it places N_{humans}
    in the room. They periodically move to other locations. They also periodically
    change the location of their objects they are holding. At the moment,
    everything is deterministic.

    """

    def __init__(self, des_size: str = "l", check_resources: bool = True) -> None:
        """Instantiate the class.

        Args:
            des_size: configuartion for the RoomDes simulation. It should be either size or
                dict. size can be "xxs (extra extra small", "xs (extra small)", "s (small)",
                "m (medium)", or "l (large)".

                {"components": <COMPONENTS>, "resources": <RESOURCES>,
                "last_timestep": <LAST_TIMESTEP>,
                "semantic_knowledge": <SEMANTIC_KNOWLEDGE>, "complexity", <COMPLEXITY>}

                <COMPONENTS> should look like this:

                <RESOURCES> should look like this:

                {'desk': 2, 'A': 10000, 'lap': 10000}

                <LAST_TIMESTEP> is a number where the DES terminates.

                <SEMANTIC_KNOWLEDGE> is a dictionary of semantic knowledge.

                <COMPLEXITY> is defined as num_humans * num_total_objects
                * maximum_num_objects_per_human * maximum_num_locations_per_object

            check_resources: whether to check if the resources are depleted or not.

        """
        if isinstance(des_size, str):
            assert des_size.lower() in [
                "dev",
                "xxs",
                "xs",
                "s",
                "m",
                "l",
            ]
            self.config = read_json(f"./data/des-config-{des_size.lower()}-v1.json")
        else:
            self.config = des_size
        self.check_resources = check_resources
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the simulator."""
        self.components = deepcopy(self.config["components"])
        self.resources = deepcopy(self.config["resources"])
        self.semantic_knowledge = [
            [key, "atlocation", val]
            for key, val in self.config["semantic_knowledge"].items()
        ]

        self.humans = []
        self.objects = []
        self.object_locations = []

        for human, obj_locs in self.components.items():
            self.humans.append(human)

            for obj, loc in obj_locs:
                self.objects.append(obj)
                self.object_locations.append(loc)

        self.humans = sorted(list(set(self.humans)))
        self.objects = sorted(list(set(self.objects)))
        self.object_locations = sorted(list(set(self.object_locations)))

        self.until = deepcopy(self.config["last_timestep"])

        self.states = []
        self.state = {}

        for human in self.components:
            self.state[human] = {}

            (
                self.state[human]["object"],
                self.state[human]["object_location"],
            ) = self.components[human][0]

            if self.check_resources:
                self.resources[self.state[human]["object_location"]] -= 1
            self.state[human]["current_time"] = 0

        self.states.append(deepcopy(self.state))
        if self.check_resources:
            for key, val in self.resources.items():
                assert val >= 0, f"{key}: {val}"

        self.events = []
        self.current_time = 0

    def step(self) -> None:
        """Proceed time by one."""
        previous_state = deepcopy(self.state)
        previous_resources = deepcopy(self.resources)

        self.current_time += 1

        for human in self.state:
            object_location_idx = self.current_time % len(self.components[human])

            self.state[human]["current_time"] = self.current_time

            if self.check_resources:
                self.resources[self.state[human]["object_location"]] += 1

            (
                self.state[human]["object"],
                self.state[human]["object_location"],
            ) = self.components[human][object_location_idx]

            if self.check_resources:
                self.resources[self.state[human]["object_location"]] -= 1

        if self.check_resources:
            for key, val in self.resources.items():
                assert val >= 0, f"{key}: {val}"

        current_state = deepcopy(self.state)
        current_resources = deepcopy(self.resources)
        self.event = self.check_event(
            previous_state, previous_resources, current_state, current_resources
        )
        self.events.append(deepcopy(self.event))
        self.states.append(deepcopy(self.state))

    def check_event(
        self,
        previous_state: dict,
        previous_resources: dict,
        current_state: dict,
        current_resources: dict,
    ) -> dict:
        """Check if any events have occured between the two consecutive states.

        Args:
            previous_state
            previous_resources
            current_state
            current_resources

        Returns:
            event

        """
        assert len(previous_state) == len(current_state)
        assert len(previous_resources) == len(current_resources)

        state_changes = {}
        resource_changes = {}

        humans = list(previous_state)
        for human in humans:
            previous_object = previous_state[human]["object"]
            previous_object_location = previous_state[human]["object_location"]
            previous_time = previous_state[human]["current_time"]

            current_object = current_state[human]["object"]
            current_object_location = current_state[human]["object_location"]
            current_time = current_state[human]["current_time"]

            assert current_time == previous_time + 1

            state_changes[human] = {}

            if previous_object != current_object:
                state_changes[human]["object"] = {
                    "previous": previous_object,
                    "current": current_object,
                }
            if previous_object_location != current_object_location:
                state_changes[human]["object_location"] = {
                    "previous": previous_object_location,
                    "current": current_object_location,
                }

            if len(state_changes[human]) == 0:
                del state_changes[human]
            else:
                state_changes[human]["current_time"] = current_time

        for resource in previous_resources:
            previous_amount = previous_resources[resource]
            current_amount = current_resources[resource]

            if previous_amount != current_amount:
                resource_changes[resource] = current_amount - previous_amount

        return {"state_changes": state_changes, "resource_changes": resource_changes}

    def run(self, debug: bool = False) -> None:
        """Run until the RoomDes terminates."""
        while self.until > 0:
            self.step()
            if debug:
                pprint(self.event)

            self.until -= 1
