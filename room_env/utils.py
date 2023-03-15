"""Utility functions"""
import json
import logging
import os
import random
import subprocess
from copy import deepcopy
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import yaml

import room_env

from .des import RoomDes

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def seed_everything(seed: int) -> None:
    """Seed every randomness to seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_lines(fname: str) -> list:
    """Read lines from a text file.

    There is some path magic going on here. This is to account for both the production
    and development mode. Don't use this function for a general purpose.

    """
    if fname.startswith("/"):
        fullpath = fname
    else:
        fullpath = os.path.join(os.path.dirname(__file__), fname)

    logging.debug(f"Reading {fullpath} ...")
    with open(fullpath, "r") as stream:
        names = stream.readlines()
    names = [line.strip() for line in names]

    return names


def read_json(fname: str) -> dict:
    """Read json"""
    logging.debug(f"reading json {fname} ...")
    with open(fname, "r") as stream:
        return json.load(stream)


def write_json(content: dict, fname: str) -> None:
    """Write json"""
    logging.debug(f"writing json {fname} ...")
    with open(fname, "w") as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_yaml(fname: str) -> dict:
    """Read yaml.

    There is some path magic going on here. This is to account for both the production
    and development mode. Don't use this function for a general purpose.

    """
    if fname.startswith("/"):
        fullpath = fname
    else:
        fullpath = os.path.join(os.path.dirname(__file__), fname)
    logging.debug(f"reading yaml {fullpath} ...")
    with open(fullpath, "r") as stream:
        return yaml.safe_load(stream)


def write_yaml(content: dict, fname: str) -> None:
    """write yaml."""
    logging.debug(f"writing yaml {fname} ...")
    with open(fname, "w") as stream:
        yaml.dump(content, stream, indent=2, sort_keys=False)


def read_data(data_path: str) -> dict:
    """Read train, val, test spilts.

    Args
    ----
    data_path: path to data.

    Returns
    -------
    data: {'train': list of training obs,
           'val': list of val obs,
           'test': list of test obs}
    """
    logging.debug(f"reading data from {data_path} ...")
    data = read_json(data_path)
    logging.info(f"Succesfully read data {data_path}")

    return data


def argmax(iterable):
    """argmax"""
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def remove_name(entity: str) -> str:
    """Remove name from the entity.

    Args
    ----
    entity: e.g., Bob's laptop

    Returns
    -------
    e.g., laptop

    """
    return entity.split()[-1]


def split_name_entity(name_entity: str) -> Tuple[str, str]:
    """Separate name and entity from the given string.

    Args
    ----
    name_entity: e.g., "Bob's laptop"

    Returns
    -------
    name: e.g., Bob
    entity: e.g., laptop

    """
    logging.debug(f"spliting name and entity from {name_entity}")
    splitted = name_entity.split()
    assert len(splitted) == 2 and "'" in splitted[0]
    name = splitted[0].split("'")[0]
    entity = splitted[1]

    return name, entity


def get_duplicate_dicts(search: dict, target: list) -> List:
    """Find if there are duplicate dicts.

    Args
    ----
    search: dict
    target: target list to look up.

    Returns
    -------
    duplicates: a list of dicts or None

    """
    assert isinstance(search, dict)
    logging.debug("finding if duplicate dicts exist ...")
    duplicates = []

    for candidate in target:
        assert isinstance(candidate, dict)
        if set(search).issubset(set(candidate)):
            if all([val == candidate[key] for key, val in search.items()]):
                duplicates.append(candidate)

    logging.info(f"{len(duplicates)} duplicates were found!")

    return duplicates


def list_duplicates_of(seq, item) -> List:
    # https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def make_des_config(
    commonsense_prob: float,
    num_humans: int,
    num_total_objects: int,
    maximum_num_objects_per_human: int,
    maximum_num_locations_per_object: int,
    maxiumum_days_period: int,
    des_size: str,
    last_timestep: int = 128,
) -> dict:
    """Make a des config.

    Args
    ----
    commonsense_prob: commonsense probability
    num_humans: number of humans
    num_total_objects: number of total objects
    maximum_num_objects_per_human: maximum number of objects per human
    maximum_num_locations_per_object: maximum number of locations per object
    maxiumum_days_period: maxiumum number of days period
    des_size: The size of DES (i.e., "xxs", "xs", "s", "m", "l", "dev")
    last_timestep: last time step where the DES terminates.

    Returns
    -------
    des config

    """
    des_config = {
        "human_names_path": "./room_env/data/human-names",
        "last_timestep": last_timestep,
        "maxiumum_days_period": maxiumum_days_period,
        "save_path": f"./room_env/data/des-config-{des_size}.json",
        "seed": 42,
        "semantic_knowledge_path": "./room_env/data/semantic-knowledge.json",
    }

    des_config["commonsense_prob"] = commonsense_prob
    des_config["num_humans"] = num_humans
    des_config["num_total_objects"] = num_total_objects
    des_config["maximum_num_objects_per_human"] = maximum_num_objects_per_human
    des_config["maximum_num_locations_per_object"] = maximum_num_locations_per_object

    return des_config


def get_des_variables(des_size: str = "l") -> Tuple[int, int, int]:
    """Get the des variables.

    Args
    ----
    des_size: The size of DES (i.e., "xxs", "xs", "s", "m", "l", "dev")

    Returns
    -------
    capacity, num_humans, num_total_objects

    """
    if des_size == "dev":
        capacity = 16

    elif des_size == "xxs":
        capacity = 2

    elif des_size == "xs":
        capacity = 4

    elif des_size == "s":
        capacity = 8

    elif des_size == "m":
        capacity = 16

    elif des_size == "l":
        capacity = 32

    else:
        raise ValueError

    num_humans = capacity * 2
    num_total_objects = capacity // 2

    return capacity, num_humans, num_total_objects


def run_des_seeds(
    seeds: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    capacity: int = 32,
    des_size: str = "l",
    allow_random_human: bool = True,
    allow_random_question: bool = True,
    question_prob: float = 0.1,
) -> dict:
    """Run the RoomEnv-v1 with multiple different seeds.

    Args
    ----
    seeds:
    capacity:
    des_size:
    allow_random_human:
    allow_random_question:
    question_prob:

    Returns
    -------
    results

    """
    results = {}
    how_to_forget = ["episodic", "semantic", "random", "pre_sem"]

    for forget_short in how_to_forget:

        if forget_short == "random":
            pretrain_semantic = False
            capacity_ = {
                "episodic": capacity // 2,
                "semantic": capacity // 2,
                "short": 1,
            }
        elif forget_short == "episodic":
            pretrain_semantic = False
            capacity_ = {"episodic": capacity, "semantic": 0, "short": 1}
        elif forget_short == "semantic":
            pretrain_semantic = False
            capacity_ = {"episodic": 0, "semantic": capacity, "short": 1}
        elif forget_short == "pre_sem":
            pretrain_semantic = True
            capacity_ = {
                "episodic": capacity // 2,
                "semantic": capacity // 2,
                "short": 1,
            }
        else:
            raise ValueError

        results_ = []
        for seed in seeds:
            env = gym.make(
                "RoomEnv-v1",
                des_size=des_size,
                seed=seed,
                policies={
                    "memory_management": "rl",
                    "question_answer": "episodic_semantic",
                    "encoding": "argmax",
                },
                capacity=capacity_,
                question_prob=question_prob,
                observation_params="perfect",
                allow_random_human=allow_random_human,
                allow_random_question=allow_random_question,
                pretrain_semantic=pretrain_semantic,
                check_resources=False,
                varying_rewards=False,
            )
            state, info = env.reset()
            rewards = 0
            while True:
                if forget_short == "random":
                    action = random.choice([0, 1, 2])
                elif forget_short == "episodic":
                    action = 0
                elif forget_short == "semantic":
                    action = 1
                elif forget_short == "pre_sem":
                    action = 0
                else:
                    raise ValueError
                state, reward, done, truncated, info = env.step(action)
                rewards += reward
                if done:
                    break
            results_.append(rewards)

        results[forget_short] = np.mean(results_)

    return results


def run_all_des_configs(
    des_size: str,
    capacity: int,
    maximum_num_objects_per_human: int,
    maximum_num_locations_per_object: int,
    maxiumum_days_period: int,
    commonsense_prob: float,
    num_humans: int,
    num_total_objects: int,
    seeds: list,
    allow_random_human: bool,
    allow_random_question: bool,
    last_timestep: int,
    question_prob: float,
) -> dict:
    """Run the RoomEnv-v1 with different des configs, with multiple different seeds.

    Args
    ----
    des_size: The size of DES (i.e., "xxs", "xs", "s", "m", "l", "dev")
    capacity: int,
    maximum_num_objects_per_human: maximum number of objects per human
    maximum_num_locations_per_object: maximum number of locations per object
    maxiumum_days_period: maxiumum number of days period
    commonsense_prob: commonsense probability
    num_humans: number of humans
    num_total_objects: number of total objects
    seeds: list,
    allow_random_human: bool,
    allow_random_question: bool,
    last_timestep: int,
    question_prob: float,

    Returns
    -------
    results

    """
    des_config = make_des_config(
        commonsense_prob=commonsense_prob,
        num_humans=num_humans,
        num_total_objects=num_total_objects,
        maximum_num_objects_per_human=maximum_num_objects_per_human,
        maximum_num_locations_per_object=maximum_num_locations_per_object,
        maxiumum_days_period=maxiumum_days_period,
        des_size=des_size,
        last_timestep=last_timestep,
    )

    complexity = (
        num_humans
        * num_total_objects
        * maximum_num_objects_per_human
        * maximum_num_locations_per_object
        * maxiumum_days_period
    )

    with open("create_des_config.yaml", "w") as stream:
        yaml.safe_dump(des_config, stream, indent=2)

    sub_out = subprocess.call(
        ["python", "create_des_config.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if sub_out == 1:
        return None

    rewards = run_des_seeds(
        seeds=seeds,
        capacity=capacity,
        des_size=des_size,
        allow_random_human=allow_random_human,
        allow_random_question=allow_random_question,
        question_prob=question_prob,
    )
    results = {
        "mean_rewards_diff": rewards["pre_sem"]
        - rewards["random"] / 3
        - rewards["semantic"] / 3
        - rewards["episodic"] / 3,
        "mean_rewards_episodic": rewards["episodic"],
        "mean_rewards_semantic": rewards["semantic"],
        "mean_rewards_random": rewards["random"],
        "mean_rewards_pre_sem": rewards["pre_sem"],
        "complexity": complexity,
        "commonsense_prob": commonsense_prob,
        "maximum_num_locations_per_object": maximum_num_locations_per_object,
        "maximum_num_objects_per_human": maximum_num_objects_per_human,
        "num_humans": num_humans,
        "num_total_objects": num_total_objects,
        "maxiumum_days_period": maxiumum_days_period,
        "allow_random_human": allow_random_human,
        "allow_random_question": allow_random_question,
        "question_prob": question_prob,
    }
    return deepcopy(results)


def fill_des_resources(des_size: str) -> None:
    """Fill resources

    Args
    ----
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


def get_handcrafted(
    env: str = "RoomEnv-v1",
    des_size: str = "l",
    seeds: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    question_prob: float = 0.1,
    observation_params: str = "perfect",
    policies: dict = {
        "memory_management": "rl",
        "question_answer": "episodic_semantic",
        "encoding": "argmax",
    },
    capacities: list = [2, 4, 8, 16, 32, 64],
    allow_random_human: bool = False,
    allow_random_question: bool = True,
    varying_rewards: bool = False,
    check_resources: bool = True,
) -> None:
    """Get the env results with handcrafted policies.

    At the moment only {"memory_management": "rl"} is supported.

    Args
    ----
    env: str = "RoomEnv-v1",
    des_size: str = "l",
    seeds: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    question_prob: float = 0.1,
    policies: dict = {
        "memory_management": "rl",
        "question_answer": "episodic_semantic",
        "encoding": "argmax",
    },
    capacities: list = [2, 4, 8, 16, 32, 64],
    allow_random_human: whether to allow random humans to be observed.
    allow_random_question: whether to allow random questions to be asked.
    varying_rewards: If true, then the rewards are scaled in every episode so that
            total_episode_rewards is 128.

    Returns
    -------
    handcrafted_results

    """
    how_to_forget = ["episodic", "semantic", "random", "pre_sem"]
    env_ = env
    handcrafted_results = {}

    for capacity in capacities:
        handcrafted_results[capacity] = {}
        for forget_short in how_to_forget:

            if forget_short == "random":
                pretrain_semantic = False
                capacity_ = {
                    "episodic": capacity // 2,
                    "semantic": capacity // 2,
                    "short": 1,
                }
            elif forget_short == "episodic":
                pretrain_semantic = False
                capacity_ = {"episodic": capacity, "semantic": 0, "short": 1}
            elif forget_short == "semantic":
                pretrain_semantic = False
                capacity_ = {"episodic": 0, "semantic": capacity, "short": 1}
            elif forget_short == "pre_sem":
                pretrain_semantic = True
                capacity_ = {
                    "episodic": capacity // 2,
                    "semantic": capacity // 2,
                    "short": 1,
                }
            else:
                raise ValueError

            results = []
            for seed in seeds:
                env = gym.make(
                    env_,
                    des_size=des_size,
                    seed=seed,
                    policies=policies,
                    capacity=capacity_,
                    question_prob=question_prob,
                    observation_params=observation_params,
                    allow_random_human=allow_random_human,
                    allow_random_question=allow_random_question,
                    pretrain_semantic=pretrain_semantic,
                    check_resources=check_resources,
                    varying_rewards=varying_rewards,
                )
                state, info = env.reset()
                rewards = 0
                while True:
                    if forget_short == "random":
                        action = random.choice([0, 1, 2])
                    elif forget_short == "episodic":
                        action = 0
                    elif forget_short == "semantic":
                        action = 1
                    elif forget_short == "pre_sem":
                        action = 0
                    else:
                        raise ValueError
                    state, reward, done, truncated, info = env.step(action)
                    rewards += reward
                    if done:
                        break
                results.append(rewards)

            mean_ = np.mean(results).round(3).item()
            std_ = np.std(results).round(3).item()
            handcrafted_results[capacity][forget_short] = {"mean": mean_, "std": std_}

    return handcrafted_results
