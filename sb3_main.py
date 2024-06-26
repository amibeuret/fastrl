import argparse
from pathlib import Path
import pprint
import random
import yaml

import uuid

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers.stats import StatsReporter, TensorboardWriter

from mlagents_envs.side_channel.stats_side_channel import (
    StatsAggregationMethod)
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel
from gym_unity.envs import UnityToGymWrapper

import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3 import PPO as PPOSB
from stable_baselines3 import SAC as SACSB
from stable_baselines3 import TD3 as TD3SB
from stable_baselines3 import DDPG, HER
from sb3_contrib import TRPO, ARS, TQC

# Added for the side channel
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage
)


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class SB3StatsRecorder(SideChannel):
    """
    Side channel that receives (string, float) pairs from the environment, so
    that they can eventually be passed to a StatsReporter.
    """

    def __init__(self,
                 stats_reporter: StatsReporter,
                 summary_freq: int = 2000) -> None:
        # >>> uuid.uuid5(uuid.NAMESPACE_URL, "com.unity.ml-agents/StatsSideChannel")
        # UUID('a1d8f7b7-cec8-50f9-b78b-d3e165a78520')
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))

        self._stats_reporter = stats_reporter
        self.summary_freq = summary_freq
        self.env_step = 0
        self.train_step = 0
        self.most_tasks_done = 0
        self.highest_score = 0
        self.last_episode_count = 0

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Receive the message from the environment, and save it for later
        retrieval.

        :param msg:
        :return:
        """
        key = msg.read_string()
        val = msg.read_float32()
        agg_type = StatsAggregationMethod(msg.read_int32())
        if agg_type == StatsAggregationMethod.AVERAGE:
            self._stats_reporter.add_stat(key, val, agg_type)
        elif agg_type == StatsAggregationMethod.SUM:
            self._stats_reporter.add_stat(key, val, agg_type)
        elif agg_type == StatsAggregationMethod.HISTOGRAM:
            self._stats_reporter.add_stat(key, val, agg_type)
        elif agg_type == StatsAggregationMethod.MOST_RECENT:

            # Hack to make sure we only get most number of tasks done for each
            # episode
            if 'NumTasks' in key:
                if val > self.most_tasks_done:
                    self.most_tasks_done = val
                    self._stats_reporter.set_stat(key, val)
            elif 'EpisodeScore' in key:
                if val > self.highest_score:
                    self.highest_score = val
                    self._stats_reporter.set_stat(key, val)
            else:
                self._stats_reporter.set_stat(key, val)
        else:
            raise NotImplemented(
                f"Unknown StatsAggregationMethod encountered. {agg_type}")

        # Another hack, otherwise the mostRecent data might be lost
        if "task" in key or "Task" in key or 'Episode' in key:
            self._stats_reporter.write_stats(self.train_step)
        elif self.train_step % self.summary_freq == 0:
            self._stats_reporter.write_stats(self.train_step)

        if 'time_step' in key:  # nice hack to sync with simulation's number
            # of steps!
            self.train_step = val

            # This for debug purposes. Is env_step different from train_step?
            # Spoiler: it is.
            self._stats_reporter.add_stat(
                'Stats/env_step', self.env_step,
                StatsAggregationMethod.AVERAGE)

        # Hack to make sure that we update the 'number of tasks' only once
        # per episode with the maximum number of tasks done.
        if 'episode_count' in key:
            if val != self.last_episode_count:
                self.last_episode_count = val
                self.most_tasks_done = 0
                self.highest_score = 0

        self.env_step = self.env_step + 1


def make_unity_env(unity_env_filename, task_name,
                   seed, base_port, env_args, no_graphics,
                   time_scale=20, summary_freq=2000, worker_id=0,
                   results_dir=None):
    # Side channels
    if results_dir is not None:
        tw = TensorboardWriter(results_dir, clear_past_data=True,
                               hidden_keys=["Is Training", "Step"])
        StatsReporter.add_writer(tw)
    stats_reporter = StatsReporter(task_name)
    stats_channel = SB3StatsRecorder(stats_reporter, summary_freq)

    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=time_scale)

    side_channels = [engine_channel, stats_channel]

    unity_env = UnityEnvironment(file_name=unity_env_filename,
                                 seed=seed,
                                 no_graphics=no_graphics,
                                 side_channels=side_channels,
                                 additional_args=env_args,
                                 base_port=base_port,
                                 worker_id=worker_id)
    env = UnityToGymWrapper(unity_env)
    return env


def run_sb3(args):
    # set all the seeds
    if isinstance(args.seed, list):
        args.seed = int(args.seed[0])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Model and training
    with open(args.ml_config_path) as file:
        sb_config = yaml.load(file, Loader=yaml.FullLoader)
        sb_args = sb_config[args.task_name]

    summary_freq = 2000
    time_scale = 20

    # Paths
    log_path = args.results_dir / args.run_id
    stats_path = log_path / "stats_reporter"
    gym_stats_path = log_path / "gym_training"

    # Create envs
    env_args = []
    env = make_unity_env(unity_env_filename=str(args.env),
                         task_name=args.task_name + "_train",
                         seed=args.seed,
                         base_port=args.initial_port,
                         env_args=env_args,
                         no_graphics=args.no_graphics,
                         time_scale=time_scale,
                         summary_freq=summary_freq,
                         results_dir=stats_path)

    # Preprocess SB args dict
    if 'policy_kwargs' in sb_args.keys():
        if isinstance(sb_args['policy_kwargs'], str):
            sb_args['policy_kwargs'] = eval(sb_args['policy_kwargs'])

    if args.algo == 'ppo':
        model = PPOSB('MlpPolicy', env, **sb_args, verbose=1,
                      tensorboard_log=str(gym_stats_path))
    elif args.algo == 'trpo':
        model = TRPO('MlpPolicy', env, **sb_args, verbose=1,
                     tensorboard_log=str(gym_stats_path))
    elif args.algo == 'ars':
        model = ARS('MlpPolicy', env, **sb_args, verbose=1,
                    tensorboard_log=str(gym_stats_path))
    elif args.algo == 'her':
        model = HER('MlpPolicy', env, **sb_args, verbose=1,
                    tensorboard_log=str(gym_stats_path))
    elif args.algo == 'tqc':
        model = TQC('MlpPolicy', env, **sb_args, verbose=1,
                    tensorboard_log=str(gym_stats_path))
    elif args.algo == 'ddpg':
        model = DDPG('MlpPolicy', env, **sb_args,
                     verbose=1, tensorboard_log=str(gym_stats_path))
    elif args.algo == 'td3':
        model = TD3SB("MlpPolicy", env, **sb_args,
                      verbose=1, tensorboard_log=str(gym_stats_path))
    elif args.algo == 'sac':
        model = SACSB("MlpPolicy", env, **sb_args, verbose=1,
                      tensorboard_log=str(gym_stats_path))
    else:
        raise NotImplemented

    eval_callback = None
    if args.evaluate:
        test_env = make_unity_env(unity_env_filename=str(args.env),
                                  task_name=args.task_name + "_test",
                                  seed=args.seed,
                                  base_port=args.initial_port + 1,
                                  env_args=env_args,
                                  no_graphics=args.no_graphics,
                                  time_scale=time_scale,
                                  summary_freq=summary_freq)
        eval_callback = EvalCallback(test_env,
                                     best_model_save_path=log_path,
                                     log_path=gym_stats_path,
                                     n_eval_episodes=1,
                                     eval_freq=10000,
                                     deterministic=True, render=False)

    new_logger = configure_logger(tensorboard_log=gym_stats_path)
    model.set_logger(new_logger)
    model.learn(total_timesteps=args.n_timesteps, callback=eval_callback)
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train agents for FAST")

    # Common arguments between mlagnets and stable baselines 3
    parser.add_argument("--env",
                        type=Path,
                        help="Path in which unity env is located")
    parser.add_argument("-n", "--run_id",
                        type=Path,
                        help="directory name for results to be saved")
    parser.add_argument("-p", "--initial_port",
                        type=int,
                        default=5005,
                        help="From this number of port + # of experiments to "
                             "run_mlagents should be a free port")
    parser.add_argument("--ml_config_path",
                        type=Path,
                        default=Path("configs/config_sb3.yaml"),
                        help="Path to the ml-agents or sb3 config. "
                             "Ex: 'configs/fast_ppo_config_linear_lr.yaml'")
    parser.add_argument("--fast_config_path",
                        type=Path,
                        default=None,
                        help="Path to the FAST config located in "
                             "StreamingAssets"
                             "Ex: 'global_custom_config.yaml'")
    parser.add_argument("--results_dir",
                        type=Path,
                        help="Path in which results of training are/will be "
                             "located")
    parser.add_argument("--seed",
                        type=int,
                        default=13,
                        help="Random seed to use. If None, different seeds "
                             "for each experiment will be used")
    parser.add_argument("--resume",
                        action='store_true',
                        help="Resume training or inference")
    parser.add_argument("--inference",
                        action='store_true',
                        help="Run inference")

    # Framework specific arguments
    parser.add_argument('--task_name', default='ImageCentering')
    parser.add_argument('--no_graphics', default=False, required=False,
                        action='store_true', help='no graphics')
    parser.add_argument('--n_envs', default=1, type=int,
                        help='number of parallel envs')
    parser.add_argument('--n_timesteps', default=30000000, type=int,
                        required=False, help='total number of steps')
    parser.add_argument("--algo",
                        default='ppo',
                        help="SB algorithm to use (ppo, sac, td3, trpo, ars, "
                             "her, tqc, ddpg)")
    parser.add_argument("--evaluate",
                        action='store_true',
                        help="Run inference")

    args = parser.parse_args()
    if args.n_envs > 1:
        raise NotImplementedError("Parallelization is not implemented")

    print("   Experiment parameters: ")
    print("-" * 100)
    pprint.pprint(vars(args), indent=5)
    print("-" * 100)

    run_sb3(args)


if __name__ == '__main__':
    main()
