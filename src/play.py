import argparse
import random
from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download
from hydra import compose, initialize
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from agent import Agent
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, collate_segments_to_batch, Dataset
from envs import make_atari_env, WorldModelEnv
from game import ActionNames, DatasetEnv, Game, get_keymap_and_action_names, Keymap, NamedEnv, PlayEnv
from headless_run import load_actions_from_json, make_noop_actions, run_headless
from utils import ATARI_100K_GAMES, get_path_agent_ckpt, prompt_atari_game


OmegaConf.register_new_resolver("eval", eval)


def download(filename: str) -> Path:
    path = hf_hub_download(repo_id="eloialonso/diamond", filename=filename)
    return Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrained", action="store_true", help="Download pretrained world model and agent.")
    parser.add_argument("-d", "--dataset-mode", action="store_true", help="Dataset visualization mode.")
    parser.add_argument("-r", "--record", action="store_true", help="Record episodes in PlayEnv.")
    parser.add_argument("-n", "--num-steps-initial-collect", type=int, default=1000, help="Num steps initial collect.")
    parser.add_argument("--store-denoising-trajectory", action="store_true", help="Save denoising steps in info.")
    parser.add_argument("--store-original-obs", action="store_true", help="Save original obs (pre resizing) in info.")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate.")
    parser.add_argument("--size", type=int, default=640, help="Window size.")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--actions", type=str, default=None, help="Path to a JSON file with an action sequence (list of action names or ints). Activates headless mode and writes a video.")
    parser.add_argument("--num-frames", type=int, default=None, help="Run headless for N additional no-op frames. Combined with --actions, pads after the JSON sequence.")
    parser.add_argument("-o", "--output", type=str, default="out.mp4", help="Output video path (headless mode only).")
    parser.add_argument("--game", type=str, default=None, choices=ATARI_100K_GAMES, help="Atari game name. Skips the interactive prompt when used with --pretrained (useful in headless mode).")
    parser.add_argument("--horizon", type=int, default=None, help="World model imagination horizon (steps before trunc). Overrides config / pretrained default (50). Image quality degrades as this grows.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for torch / numpy / random. Reproduces initial conditioning batch + diffusion noise + reward/end sampling.")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.dataset_mode:
        if not Path("dataset").is_dir():
            print(f"Error: {str(Path('dataset').absolute())} not found, cannot use dataset mode.")
            return False
        if Path(".git").is_dir():
            print("Error: cannot run dataset mode the root of the repository.")
            return False
        if args.pretrained or args.record:
            print("Warning: dataset mode, ignoring --pretrained and --record")
    else:
        if not args.record and (args.store_denoising_trajectory or args.store_original_obs):
            print("Warning: not in recording mode, ignoring --store* options")
    return True


def prepare_dataset_mode(cfg: DictConfig) -> Tuple[DatasetEnv, Keymap, ActionNames]:
    datasets = []
    for p in Path("dataset").iterdir():
        if p.is_dir():
            d = Dataset(p, p.stem)
            d.load_from_default_path()
            datasets.append(d)
    _, env_action_names = get_keymap_and_action_names(cfg.env.keymap)
    dataset_env = DatasetEnv(datasets, env_action_names)
    keymap, _ = get_keymap_and_action_names("dataset_mode")
    return dataset_env, keymap


def prepare_play_mode(cfg: DictConfig, args: argparse.Namespace) -> Tuple[PlayEnv, Keymap, ActionNames]:
    # Checkpoint
    if args.pretrained:
        name = args.game if args.game is not None else prompt_atari_game()
        path_ckpt = download(f"atari_100k/models/{name}.pt")

        # Override config
        cfg.agent = OmegaConf.load(download("atari_100k/config/agent/default.yaml"))
        cfg.env = OmegaConf.load(download("atari_100k/config/env/atari.yaml"))
        cfg.env.train.id = cfg.env.test.id = f"{name}NoFrameskip-v4"
        cfg.world_model_env.horizon = 50
    else:
        path_ckpt = get_path_agent_ckpt("checkpoints", epoch=-1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Real envs
    train_env = make_atari_env(num_envs=1, device=device, **cfg.env.train)
    test_env = make_atari_env(num_envs=1, device=device, **cfg.env.test)

    # Models
    agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
    agent.load(path_ckpt)

    # Collect for imagination's initialization
    n = args.num_steps_initial_collect
    dataset = Dataset(Path(f"dataset/{path_ckpt.stem}_{n}"))
    dataset.load_from_default_path()
    if len(dataset) == 0:
        print(f"Collecting {n} steps in real environment for world model initialization.")
        collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=0)
        collector.send(NumToCollect(steps=n))
        dataset.save_to_default_path()

    # World model environment
    bs = BatchSampler(dataset, 0, 1, 1, cfg.agent.denoiser.inner_model.num_steps_conditioning, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg, return_denoising_trajectory=True)

    envs = [
        NamedEnv("wm", wm_env),
        NamedEnv("test", test_env),
        NamedEnv("train", train_env),
    ]

    env_keymap, env_action_names = get_keymap_and_action_names(cfg.env.keymap)
    play_env = PlayEnv(
        agent,
        envs,
        env_action_names,
        env_keymap,
        args.record,
        args.store_denoising_trajectory,
        args.store_original_obs,
    )

    return play_env, env_keymap


@torch.no_grad()
def main():
    args = parse_args()
    ok = check_args(args)
    if not ok:
        return

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"Seeded RNGs with {args.seed}")

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")

    env, keymap = prepare_dataset_mode(cfg) if args.dataset_mode else prepare_play_mode(cfg, args)
    size = (args.size // cfg.env.train.size) * cfg.env.train.size  # window size

    headless = args.actions is not None or args.num_frames is not None
    if headless:
        if args.dataset_mode:
            print("Error: headless mode (--actions / --num-frames) is not supported in --dataset-mode.")
            return
        actions: list = []
        if args.actions is not None:
            actions.extend(load_actions_from_json(Path(args.actions), env.action_names))
        if args.num_frames is not None:
            actions.extend(make_noop_actions(args.num_frames))

        wm_env = next((e for name, e in env.envs if name == "wm"), None)
        if wm_env is not None:
            desired_horizon = args.horizon if args.horizon is not None else len(actions) + 10
            if wm_env.horizon < desired_horizon:
                print(f"Bumping world-model horizon: {wm_env.horizon} -> {desired_horizon}")
                wm_env.horizon = desired_horizon

        run_headless(env, actions, Path(args.output), fps=args.fps, size_hw=(size, size))
        return

    if args.horizon is not None:
        wm_env = next((e for name, e in env.envs if name == "wm"), None)
        if wm_env is not None:
            wm_env.horizon = args.horizon

    game = Game(env, keymap, (size, size), fps=args.fps, verbose=not args.no_header)
    game.run()


if __name__ == "__main__":
    main()
