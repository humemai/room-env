import json
import logging
import os
import random
from copy import deepcopy
from typing import List

import yaml

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_objects_and_locations(
    total_objects: list,
    maximum_num_objects_per_human: int,
    maxiumum_days_period: int,
    maximum_num_locations_per_object: int,
    commonsense_prob: float,
    possible_object_locations: list,
    semantic_knowledge: dict,
) -> List:
    """Get objects and their locations for one human.

    Args
    ----
    total_objects: total possible objects.
    maximum_num_objects_per_human: maximum number of objects per human
    maxiumum_days_period: maximum number of days per period.
    maximum_num_locations_per_object: maximum number of locations per object.
    commonsense_prob: the probability of an object being located at a commonsense
        location.
    possible_object_locations: possible object locations,
    semantic_knowledge: commonsense knowledge,

    Returns
    -------
    objects and their locations (e.g., [["laptop", "desk"], ["laptop", "desk"],
        ["laptop", "table"], ["laptop", "desk"]])

    """
    logging.debug("Getting objects and their locations for one human ...")

    random.shuffle(total_objects)
    num_objects = random.randint(1, maximum_num_objects_per_human)
    objs = total_objects[:num_objects]

    object_locations = []

    for obj in objs:
        num_days_period = random.randint(1, maxiumum_days_period)
        num_locations_per_object = random.randint(1, maximum_num_locations_per_object)

        probs = [commonsense_prob] + [
            (1 - commonsense_prob) / (num_locations_per_object - 1)
            for _ in range(num_locations_per_object - 1)
        ]

        obj_locs = [semantic_knowledge[obj]]
        count = 0
        while len(obj_locs) != num_locations_per_object:
            # print(obj_locs)

            loc = random.choice(possible_object_locations)
            if loc not in obj_locs:
                obj_locs.append(loc)
            count += 1

        assert len(obj_locs) == len(probs)

        obj_locs = random.choices(obj_locs, probs, k=num_days_period)
        for loc in obj_locs:
            object_locations.append([obj, loc])

    return object_locations


def main(
    semantic_knowledge_path: str,
    human_names_path: str,
    save_path: str,
    num_humans: int,
    num_total_objects: int,
    maximum_num_objects_per_human: int,
    maximum_num_locations_per_object: int,
    commonsense_prob: float,
    maxiumum_days_period: int,
    last_timestep: int,
    seed: int,
) -> None:
    """Run!

    Args
    ----
    semantic_knowledge_path: e.g., "./room_env/data/semantic-knowledge.json"
    human_names_path:  e.g., "./room_env/data/human-names"
    save_path: e.g., "./room_env/data/des-config-m.json"
    num_humans: e.g., 8
    num_total_objects: e.g., 8
    maximum_num_locations_per_object: maximum number of locations per object (e.g., 8)
    commonsense_prob: commonsense probability (e.g., 0.8)
    maxiumum_days_period: maximum number of days per period.
    last_timestep: the last day when the DES stops (e.g., 1000).
    seed: random seed

    """
    assert num_total_objects >= maximum_num_objects_per_human

    config = {"components": {}, "resources": {}, "last_timestep": last_timestep}

    assert maximum_num_locations_per_object <= maxiumum_days_period

    # for reproducibility
    random.seed(seed)
    with open(human_names_path, "r") as stream:
        human_names = [foo.strip() for foo in stream.readlines()]

    with open(semantic_knowledge_path, "r") as stream:
        semantic_knowledge = json.load(stream)

    assert num_humans <= len(human_names)

    logging.debug(
        f"There were {len(semantic_knowledge)} objects before removing the duplicate "
        "object locations."
    )
    unique_locations = []

    for key, val in deepcopy(semantic_knowledge).items():
        if "_" in key:
            del semantic_knowledge[key]
            continue
        if val["AtLocation"][0]["tail"] in unique_locations:
            del semantic_knowledge[key]
            continue
        if "_" in val["AtLocation"][0]["tail"]:
            del semantic_knowledge[key]
            continue
        # This avoids locations being same as object names.
        if val["AtLocation"][0]["tail"] in list(semantic_knowledge):
            del semantic_knowledge[key]
            continue

        unique_locations.append(val["AtLocation"][0]["tail"])

    logging.info(
        f"There are now {len(semantic_knowledge)} objects before after the duplicate "
        "object locations."
    )

    semantic_knowledge = {
        key: val["AtLocation"][0]["tail"] for key, val in semantic_knowledge.items()
    }

    assert num_total_objects <= len(semantic_knowledge)
    assert maximum_num_objects_per_human <= len(semantic_knowledge)
    assert maximum_num_locations_per_object <= len(semantic_knowledge)

    random.shuffle(human_names)
    total_humans = human_names[:num_humans]

    total_objects = list(semantic_knowledge)
    random.shuffle(total_objects)
    total_objects = total_objects[:num_total_objects]

    possible_object_locations = list(semantic_knowledge.values())
    possible_object_locations = [
        loc
        for loc in possible_object_locations
        if loc not in human_names and loc not in total_objects
    ]

    semantic_knowledge = {obj: semantic_knowledge[obj] for obj in total_objects}

    assert len(semantic_knowledge) == num_total_objects

    random.shuffle(total_humans)

    for human in total_humans:

        config["components"][human] = get_objects_and_locations(
            total_objects,
            maximum_num_objects_per_human,
            maxiumum_days_period,
            maximum_num_locations_per_object,
            commonsense_prob,
            possible_object_locations,
            semantic_knowledge,
        )

    config["semantic_knowledge"] = semantic_knowledge

    config["resources"] = {}
    config["complexity"] = (
        num_humans
        * num_total_objects
        * maximum_num_objects_per_human
        * maximum_num_locations_per_object
        * maxiumum_days_period
    )
    with open(save_path, "w") as stream:
        json.dump(config, stream, indent=4, sort_keys=False)

    logging.info(f"DES config done! they are saved at {save_path}")


if __name__ == "__main__":
    with open("./create_des_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    print("Arguments:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(**config)
