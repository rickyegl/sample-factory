from typing import Optional

import gymnasium as gym

from sample_factory.envs.env_wrappers import (
    ClipRewardEnv,
    NESEpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    NumpyObsWrapper,
)

#Change
ATARI_W = ATARI_H = 84


class NESSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


NES_ENVS = [
    NESSpec("mario", "gym_super_mario_bros:SuperMarioBros-v0"),
    NESSpec("marioR", "gym_super_mario_bros:SuperMarioBrosRandomStages-v0"),
]


def atari_env_by_name(name):
    for cfg in NES_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown NES env")


def make_nes_env(env_name, cfg, env_config, render_mode: Optional[str] = None):
    atari_spec = atari_env_by_name(env_name)
    print("Making " + atari_spec.env_id + " from " + env_name)
    env = gym.make(atari_spec.env_id)#, render_mode=render_mode)

    if atari_spec.default_timeout is not None:
        env._max_episode_steps = atari_spec.default_timeout

    # these are chosen to match Stable-Baselines3 and CleanRL implementations as precisely as possible
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=cfg.env_frameskip)
    env = NESEpisodicLifeEnv(env)
    # noinspection PyUnresolvedReferences
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, cfg.env_framestack)
    env = NumpyObsWrapper(env)
    return env
