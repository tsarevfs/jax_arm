# %load_ext autoreload
#
# %autoreload 2

import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import functools
import json
from datetime import datetime

import jax
import mediapy as media
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
# from IPython.display import clear_output, display # Not needed for script
from orbax import checkpoint as ocp

from mujoco_playground import wrapper, manipulation
from mujoco_playground.config import manipulation_params

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

env_name = "PandaPickCubeCartesian"
env_cfg = manipulation.get_default_config(env_name)

SUFFIX = None
FINETUNE_PATH = None

# Generate unique experiment name.
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
exp_name = f"{env_name}/{timestamp}"
if SUFFIX is not None:
  exp_name += f"-{SUFFIX}"
print(f"Experiment name: {exp_name}")

wandb.init(
    project="jax_arm",
    name=exp_name,
    config=env_cfg.to_dict(),
)

ckpt_path = epath.Path("checkpoints").resolve() / exp_name
ckpt_path.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint path: {ckpt_path}")

with open(ckpt_path / "config.json", "w") as fp:
  json.dump(env_cfg.to_dict(), fp, indent=4)

def progress(num_steps, metrics):
  times.append(datetime.now())
  x_data.append(num_steps)

  fps = 0
  if len(x_data) >= 2:
    num = x_data[-1] - x_data[-2]
    denom = (times[-1] - times[-2]).total_seconds()
    if denom > 0:
      fps = num / denom

    wandb.log({
        "perf/fps": fps,
    }, step=num_steps)

  wandb.log(metrics, step=num_steps)

def policy_params_fn(current_step, make_policy, params):
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  path = ckpt_path / f"{current_step}"
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)

train_params = dict(manipulation_params.brax_ppo_config(env_name))
train_params['seed'] = 1
train_params['num_evals'] = 30

train_params["network_factory"] = functools.partial(
    ppo_networks.make_ppo_networks,
    **dict(train_params["network_factory"])
  )

train_params["progress_fn"] = progress
train_params["training_metrics_steps"] = 10000
train_params["log_training_metrics"] = True

train_fn = functools.partial(ppo.train, **train_params)

times = [datetime.now()]
x_data = []

env = manipulation.load(env_name, config=env_cfg)
make_inference_fn, params, _ = train_fn(
    wrap_env=False,
    environment=wrapper.wrap_for_brax_training(env,
                                               episode_length=env_cfg.episode_length,
                                               action_repeat=env_cfg.action_repeat),

)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

env = manipulation.load(env_name, config=env_cfg)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

key = jax.random.PRNGKey(5)
key, key_reset = jax.random.split(key)
state = jit_reset(key_reset)
states = [state]

render_every = 2  # Policy is 50 FPS

for i in range(125):
  act_rng, key = jax.random.split(key)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  if i % render_every == 0: 
    states.append(state)

# In a script, we write the video to a file.
video = env.render(states, height=480, width=640)
media.write_video("output.mp4", video, fps=1.0 / env.dt / render_every)
print("Video saved to output.mp4")
