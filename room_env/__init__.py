from gymnasium.envs.registration import register

register(
    id="RoomEnv-v2",
    entry_point="room_env.envs:RoomEnv2",
)
