# Can an RL agent learn a more complicated environment?

## [Now the environment has more than one room.](./Homework%20from%20Michael_221018_180939.jpg)

- This adds one more action space. That is, the agent has to decide which room to go at time t. In that room, the initial action space follows, where it has to decide in which memory system it should store its observation.

## Observations can also be extended.

- Instead of simple triples, (one’s object, AtLocation, furniture), it can include more things.
- This means that the questions will also become more complicated. Instead of simply asking the object location, it can also ask other things.

## Delayed rewards

- Currently, a question is asked every time the agent takes an action. This is not realistic. Questions can be asked at any time, leading to delayed rewards.
- Actually, this is already done. Delayed rewards work fine.

## Other things to consider

- LSTM might not be a suitable function apprximator any more here, but a more generic GNN might be a better option.
- DQN still might work well, but other RL algos might work better.
