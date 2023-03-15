# The Room environment - v0

[There is a newer version, v1](../README.md)

We have released a challenging [Gymnasium](https://www.gymlibrary.dev/) compatible
environment. The best strategy for this environment is to have both episodic and semantic
memory systems. See the [paper](https://arxiv.org/abs/2204.01611) for more information.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.8 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.
1. This env is added to the PyPI server. Just run: `pip install room-env`

## Data collection

Data is collected from querying ConceptNet APIs. For simplicity, we only collect triples
whose format is (`head`, `AtLocation`, `tail`). Here `head` is one of the 80 MS COCO
dataset categories. This was kept in mind so that later on we can use images as well.

If you want to collect the data manually, then run below:

```
python collect_data.py
```

## How does this environment work?

The Gymnasium-compatible Room environment is one big room with
_N_<sub>_people_</sub> number of people who can freely move
around. Each of them selects one object, among
_N_<sub>_objects_</sub>, and places it in one of the
_N_<sub>_locations_</sub> locations.
_N_<sub>_agents_</sub> number of agent(s) are also in this
room. They can only observe one human placing an object, one at a time;
**x**<sup>(_t_)</sup>. At the same time, they are given one question
about the location of an object; **q**<sup>(_t_)</sup>.
**x**<sup>(_t_)</sup> is given as a quadruple,
(**h**<sup>(_t_)</sup>,**r**<sup>(_t_)</sup>,**t**<sup>(_t_)</sup>,_t_),
For example, `<James’s laptop, AtLocation, James’s desk, 42>` accounts
for an observation where an agent sees James placing his laptop on his
desk at *t* = 42. **q**<sup>(_t_)</sup> is given as a double,
(**h**,**r**). For example, `<Karen’s cat, AtLocation>` is asking where
Karen’s cat is located. If the agent answers the question correctly, it
gets a reward of  + 1, and if not, it gets 0.

The reason why the observations and questions are given as
RDF-triple-like format is two folds. One is that this structured format
is easily readable / writable by both humans and machines. Second is
that we can use existing knowledge graphs, such as ConceptNet .

To simplify the environment, the agents themselves are not actually
moving, but the room is continuously changing. There are several random
factors in this environment to be considered:

1. With the chance of _p_<sub>commonsense</sub>,
   a human places an object in a commonsense location (e.g., a laptop
   on a desk). The commonsense knowledge we use is from ConceptNet.
   With the chance of
   1 − *p*<sub>_commonsense_</sub>, an object is
   placed at a non-commonsense random location (e.g., a laptop on the
   tree).

1. With the chance of
   _p_<sub>_new_\__location_</sub>, a human changes
   object location.

1. With the chance of _p_<sub>_new_\__object_</sub>, a
   human changes his/her object to another one.

1. With the chance of
   _p_<sub>_switch_\__person_</sub>, two people
   switch their locations. This is done to mimic an agent moving around
   the room.

All of the four probabilities account for the Bernoulli distributions.

Consider there is only one agent. Then this is a POMDP, where _S_<sub>_t_</sub> = (**x**<sup>(_t_)</sup>, **q**<sup>(_t_)</sup>), _A_<sub>_t_</sub> = (do something with **x**<sup>(_t_)</sup>, answer **q**<sup>(_t_)</sup>), and _R_<sub>_t_</sub> ∈ *{0, 1}*.

Currently there is no RL trained for this. We only have some heuristics. Take a look at the paper for more details.

## RoomEnv-v0

```python
import gymnasium as gym
import room_env

env = gym.make("RoomEnv-v0")
(observation, question), info = env.reset()
rewards = 0

while True:
    (observation, question), reward, done, truncated, info = env.step("This is my answer!")
    rewards += reward
    if done:
        break

print(rewards)
```

Every time when an agent takes an action, the environment will give you an observation
and a question to answer. You can try directly answering the question,
such as `env.step("This is my answer!")`, but a better strategy is to keep the
observations in memory systems and take advantage of the current observation and the
history of them in the memory systems.

Take a look at [this repo](https://github.com/tae898/explicit-memory) for an actual
interaction with this environment to learn a policy.

## Contributing

Contributions are what make the open source community such an amazing place to be learn,
inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory,
   to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## [Cite our paper](https://arxiv.org/abs/2204.01611)

```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.01611,
  doi = {10.48550/ARXIV.2204.01611},
  url = {https://arxiv.org/abs/2204.01611},
  author = {Kim, Taewoon and Cochez, Michael and Francois-Lavet, Vincent and Neerincx,
  Mark and Vossen, Piek},
  keywords = {Artificial Intelligence (cs.AI), FOS: Computer and information sciences,
  FOS: Computer and information sciences},
  title = {A Machine With Human-Like Memory Systems},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Cite our code

[![DOI](https://zenodo.org/badge/477781069.svg)](https://zenodo.org/badge/latestdoi/477781069)

## Authors

- [Taewoon Kim](https://taewoon.kim/)
- [Michael Cochez](https://www.cochez.nl/)
- [Vincent Francois-Lavet](http://vincent.francois-l.be/)
- [Mark Neerincx](https://ocw.tudelft.nl/teachers/m_a_neerincx/)
- [Piek Vossen](https://vossen.info/)

## License

[MIT](https://choosealicense.com/licenses/mit/)
