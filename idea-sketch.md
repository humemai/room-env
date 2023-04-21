# [Capturing dynamic knowledge graphs with human-like memory systems by reinforcement learning](https://www.overleaf.com/9972424828hxktdqjxnmsw)

## The three policies

1. Question answering policy: $\pi_{qa}(a_{qa}|M_{long})$
2. Memory management policy: $\pi_{memory}(a_{memory} | M_{short}, M_{long})$
3. Exploration policy: $\pi_{explore}(a_{explore} | M_{long})$

First, I'll have the agent learn one policy, while the other two are fixed. How can I do
this actually? Should this be one environment?

## The knowledge graphs

- Both the hidden states and the observed sub-graphs are knowledge graphs but they are
  still quite too simple to be called a knowledge graph. They only have one relation.
  How can I add more relations?
- 