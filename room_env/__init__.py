from gymnasium.envs.registration import register

register(
    id="RoomEnv-v0",
    entry_point="room_env.envs:RoomEnv0",
)

register(
    id="RoomEnv-v1",
    entry_point="room_env.envs:RoomEnv1",
)

register(
    id="RoomEnv-v2",
    entry_point="room_env.envs:RoomEnv2",
)
