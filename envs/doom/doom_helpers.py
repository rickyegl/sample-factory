import cv2
import numpy as np


def key_to_action(key):
    from pynput.keyboard import Key

    action_table = {
        Key.up: 0,
        Key.down: 1,
        Key.right: 4,
        Key.left: 5,
        Key.ctrl: 6,
        Key.shift: 7,
    }

    return action_table.get(key, None)


_DOOM_WINDOWS = set()


def concat_grid(obs):
    obs = [cvt_doom_obs(o) for o in obs]

    max_horizontal = 4
    horizontal = min(max_horizontal, len(obs))
    vertical = max(1, len(obs) // horizontal)

    assert vertical * horizontal == len(obs)

    obs_concat_h = []
    for i in range(vertical):
        obs_concat_h.append(np.concatenate(obs[i * horizontal:(i + 1) * horizontal], axis=1))

    obs_concat = np.concatenate(obs_concat_h, axis=0)
    return obs_concat


def cvt_doom_obs(obs):
    w, h = 960, 540
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    obs = cv2.resize(obs, (w, h))
    return obs
