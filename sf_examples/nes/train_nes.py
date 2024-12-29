import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.atari.atari_params import atari_override_defaults
from sf_examples.nes.nes_utils import NES_ENVS, make_nes_env


def register_atari_envs():
    for env in NES_ENVS:
        register_env(env.name, make_nes_env)


def register_atari_components():
    register_atari_envs()


def parse_atari_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    atari_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_atari_components()
    cfg = parse_atari_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
