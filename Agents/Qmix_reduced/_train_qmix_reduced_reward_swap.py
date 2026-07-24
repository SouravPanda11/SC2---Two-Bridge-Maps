"""
Isolated QMIX driver for the terminal-success-reward swap experiment.

The established per-map QMIX trainer is reused without modifying it. A
spawn-safe parallel environment batch loads only the new reward-swap
environment modules in worker processes.
"""

import atexit
import importlib
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import traceback

import numpy as np


AGENT_NAME = "QMIX_reduced_terminal_reward_swap"

TRAIN_MODULES = {
    "V1_Base": "Agents.Qmix_reduced.V1_Base.train_qmix",
    "V2_Navigate": "Agents.Qmix_reduced.V2_Navigate.train_qmix",
}

MAP_ENV_MODULES = {
    "V1_Base": (
        "Environments.QMIX_reduced."
        "TB_env_QMIX_reduced_reward_swap_V1_Base"
    ),
    "V2_Navigate": (
        "Environments.QMIX_reduced."
        "TB_env_QMIX_reduced_reward_swap_V2_Navigate"
    ),
}


def _load_env_class(map_name: str):
    module = importlib.import_module(MAP_ENV_MODULES[map_name])
    return module.TwoBridgeEnv


def _make_env_payload(env, obs):
    return {
        "obs": np.asarray(obs, dtype=np.float32),
        "state": np.asarray(env.get_state(), dtype=np.float32),
        "minimap": np.asarray(env.get_minimap(), dtype=np.uint8),
        "avail_actions": np.asarray(
            env.get_avail_actions(),
            dtype=np.float32,
        ),
    }


def qmix_reward_swap_env_worker(
    remote,
    parent_remote,
    rank: int,
    env_module: str,
    env_kwargs: dict,
):
    parent_remote.close()
    worker_tmp_dir = tempfile.mkdtemp(
        prefix=f"tbm-qmix-reward-swap-worker-{rank}-"
    )
    cleanup = lambda path=worker_tmp_dir: shutil.rmtree(
        path,
        ignore_errors=True,
    )
    atexit.register(cleanup)

    try:
        for key in ("TMP", "TEMP", "TMPDIR"):
            os.environ[key] = worker_tmp_dir

        time.sleep(0.5 * rank)
        env_cls = importlib.import_module(env_module).TwoBridgeEnv
        env = env_cls(**env_kwargs)

        while True:
            cmd, data = remote.recv()
            if cmd == "get_env_info":
                remote.send(env.get_env_info())
            elif cmd == "reset":
                obs, _ = env.reset()
                remote.send(_make_env_payload(env, obs))
            elif cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                payload = _make_env_payload(env, obs)
                payload.update(
                    {
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": info,
                    }
                )
                remote.send(payload)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise ValueError(f"Unknown worker command: {cmd!r}")
    except EOFError:
        pass
    except Exception:
        error_message = traceback.format_exc()
        try:
            remote.send({"__worker_error__": error_message})
        except Exception:
            pass
    finally:
        try:
            remote.close()
        except Exception:
            pass
        cleanup()


def _make_parallel_env_batch(base_module):
    class ParallelQMixRewardSwapEnvBatch(
        base_module.ParallelQMixEnvBatch
    ):
        def __init__(
            self,
            num_envs: int,
            base_seed: int,
            env_kwargs: dict,
        ):
            self.num_envs = int(num_envs)
            self.closed = False
            self.ctx = mp.get_context("spawn")
            self.remotes = []
            self.processes = []

            map_name = str(env_kwargs["map_name"])
            env_module = MAP_ENV_MODULES[map_name]

            for rank in range(self.num_envs):
                parent_remote, worker_remote = self.ctx.Pipe()
                worker_kwargs = dict(env_kwargs)
                worker_kwargs["seed"] = int(base_seed + rank)
                process = self.ctx.Process(
                    target=qmix_reward_swap_env_worker,
                    args=(
                        worker_remote,
                        parent_remote,
                        rank,
                        env_module,
                        worker_kwargs,
                    ),
                )
                process.daemon = True
                process.start()
                worker_remote.close()
                self.remotes.append(parent_remote)
                self.processes.append(process)

            self.env_info = self.call(0, "get_env_info")

    return ParallelQMixRewardSwapEnvBatch


def train_with_settings(
    *,
    map_name: str,
    seed_values: tuple[int, ...],
    total_timesteps: int = 2_000_000,
    num_seeds: int = 2,
    num_envs: int = 3,
    save_interval: int = 50_000,
    run_mode: str = "fresh_start",
):
    if map_name not in TRAIN_MODULES:
        known = ", ".join(sorted(TRAIN_MODULES))
        raise ValueError(
            f"Reward-swap QMIX supports only {known}; received {map_name!r}."
        )
    if len(seed_values) != num_seeds:
        raise ValueError(
            "seed_values must contain exactly num_seeds entries; received "
            f"{len(seed_values)} values for num_seeds={num_seeds}."
        )

    base_module = importlib.import_module(TRAIN_MODULES[map_name])
    env_class = _load_env_class(map_name)
    parallel_env_batch = _make_parallel_env_batch(base_module)

    replacements = {
        "TwoBridgeEnv": env_class,
        "ParallelQMixEnvBatch": parallel_env_batch,
        "AGENT_NAME": AGENT_NAME,
        "MAP_NAME": map_name,
        "RUN_MODE": run_mode,
        "FRESH_START_SEED": None,
        "FRESH_START_SEED_VALUES": tuple(int(seed) for seed in seed_values),
        "TOTAL_TIMESTEPS": int(total_timesteps),
        "SAVE_INTERVAL": int(save_interval),
        "NUM_SEEDS": int(num_seeds),
        "NUM_ENVS": int(num_envs),
    }
    originals = {
        name: getattr(base_module, name)
        for name in replacements
    }

    for name, value in replacements.items():
        setattr(base_module, name, value)

    try:
        config = base_module.build_run_config()
        return base_module.train(config)
    finally:
        for name, value in originals.items():
            setattr(base_module, name, value)
