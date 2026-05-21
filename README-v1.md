# The Room environment - v1

This environment is part of [A Machine with Short-Term, Episodic, and Semantic Memory
Systems](https://humem.ai/projects/explicit-memory). Check that project page for the
paper and more context.

We have released a challenging [Gymnasium](https://www.gymlibrary.dev/) compatible
environment. The best strategy for this environment is to have both episodic and semantic
memory systems.

## Prerequisites

1. python 3.10 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly
   recommended so that you don't mess up with the system python.
1. This env is added to the PyPI server. Just run: `pip install room-env`

## RoomEnv-v1

```python
import gymnasium as gym

env = gym.make("room_env:RoomEnv-v1")
(observation, question), info = env.reset()
rewards = 0

while True:
    # There is one different thing in the RoomEnv from the original AAAI-2023 paper:
    # The reward is either +1 or -1, instead of +1 or 0.
    (observation, question), reward, done, truncated, info = env.step("This is my answer!")
    rewards += reward
    if done:
        break
```

```json
{
    "des_size": "l",
    "seed": 42,
    "question_prob": 1.0,
    "allow_random_human": False,
    "allow_random_question": False,
    "total_maximum_episode_rewards": 128,
    "check_resources": True,
}
```

If you want to create an env with a different set of parameters, you can do so. For example:

```python
env_params = {"seed": 0,
              "allow_random_human": True,
              "pretrain_semantic": True}
env = gym.make("room_env:RoomEnv-v1", **env_params)
```

Take a look at [this repo](https://github.com/humemai/humemai) for an actual
interaction with this environment to learn a policy.

## Data collection

Data is collected from querying ConceptNet APIs. For simplicity, we only collect triples
whose format is (`head`, `atlocation`, `tail`). Here `head` is one of the 80 MS COCO
dataset categories. This was kept in mind so that later on we can use images as well.

If you want to collect the data manually, then run below:

```
python collect_data.py
```

## [The RoomDes](room_env/des.py)

The DES is part of RoomEnv. You don't have to care about how it works. If you are still
curious, you can read below.

You can run the RoomDes by

```python
from room_env.des import RoomDes

des = RoomDes()
des.run(debug=True)
```

with `debug=True` it'll print events (i.e., state changes) to the console.

```console
{'resource_changes': {'desk': -1, 'lap': 1},
 'state_changes': {'Vincent': {'current_time': 1,
                               'object_location': {'current': 'desk',
                                                   'previous': 'lap'}}}}
{'resource_changes': {}, 'state_changes': {}}
{'resource_changes': {}, 'state_changes': {}}
{'resource_changes': {},
 'state_changes': {'Michael': {'current_time': 4,
                               'object_location': {'current': 'lap',
                                                   'previous': 'desk'}},
                   'Tae': {'current_time': 4,
                           'object_location': {'current': 'desk',
                                               'previous': 'lap'}}}}
```

## Authors

- [Taewoon Kim](https://taewoon.kim/)
- [Michael Cochez](https://www.cochez.nl/)
- [Vincent Francois-Lavet](http://vincent.francois-l.be/)
- [Mark Neerincx](https://ocw.tudelft.nl/teachers/m_a_neerincx/)
- [Piek Vossen](https://vossen.info/)

## License

[MIT](https://choosealicense.com/licenses/mit/)
