# [Capturing dynamic knowledge graphs with human-like memory systems by reinforcement learning](https://www.overleaf.com/9972424828hxktdqjxnmsw)

## Now the environment has more than one room.

- This adds one more action space. That is, the agent has to decide which room to go at
  time $t$. In that room, the initial action space follows, where it has to decide in which
  memory system it should store its observation.

## Observations can also be extended.

- Instead of simple triples, (one’s object, AtLocation, furniture), it can include more things.
- This means that the questions will also become more complicated. Instead of simply asking the object location, it can also ask other things.

## Delayed rewards

- Currently, a question is asked every time the agent takes an action. This is not realistic. Questions can be asked at any time, leading to delayed rewards.
- Actually, this is already done. Delayed rewards work fine.

## Other things to consider

- LSTM might not be a suitable function apprximator any more here, but a more generic GNN might be a better option.
- DQN still might work well, but other RL algos might work better.

## The three policies

1. Question answering policy: $\pi_{qa}(a_{qa}|M_{long})$
1. Memory management policy: $\pi_{memory}(a_{memory} | M_{short}, M_{long})$
1. Exploration policy: $\pi_{explore}(a_{explore} | M_{long})$

First, I'll have the agent learn one policy, while the other two are fixed. How can I do
this actually? Should this be one environment?

## The knowledge graphs

- Both the hidden states and the observed sub-graphs are knowledge graphs but they are
  still quite too simple to be called a knowledge graph. They only have one relation.
  How can I add more relations?
