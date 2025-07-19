# Room Environment v3

A simplified version of the Room Environment with deterministic object movement, periodic inner walls, and fixed question count.

## Key Features

- **Square Grid**: Always `grid_length²` rooms in a square layout
- **Deterministic Movement**: Objects move in predetermined patterns
- **Periodic Inner Walls**: Selected walls turn on/off in variable-period cycles
- **Fixed 100 Questions**: Always exactly 100 questions, one per step
- **Flexible Episodes**: Any episode length with cycling behavior

## Creating an Environment

```python
from room_env.create_room_v3 import RoomCreator

room_creator = RoomCreator(
    filename="dev",
    grid_length=3,
    num_static_objects=3,
    num_moving_objects=3,
    num_inner_walls=3,  # Number of walls to select for periodic behavior
    seed=42,
)
room_creator.run()
```

## Using the Environment

```python
from room_env.envs.room3 import RoomEnv3

env_config = {
    "terminates_at": 99,  # Can be any value
    "room_size": "small",
}

env = RoomEnv3(**env_config)
observations, info = env.reset()

while True:
    # Single answer for single question
    question_answer = "room_name"  # Single string answer
    movement_action = "north"  # or "east", "south", "west", "stay"
    
    observations, reward, done, truncated, info = env.step((question_answer, movement_action))
    
    if done:
        break
```

## Configuration

- **terminates_at**: Episode length (can be any value)
- **room_size**: Configuration name
- **num_questions**: Ignored - always 100 questions
- **new_wall_layout_interval**: Ignored - walls follow periodic patterns

## Periodic Behavior

### Questions
- Always exactly 100 predefined questions
- One question per step
- Questions cycle if episode length > 100

### Inner Walls
- Selected walls follow variable-period on/off patterns
- Each wall assigned a random pattern from the available set
- Patterns cycle based on their individual lengths

### Object Movement
- **Static objects**: Never move
- **Moving objects**: Follow predetermined direction preferences
- **Agent**: Controlled by actions

The agent starts in the center room and gets +1 reward for correct answers, 0 for wrong answers.