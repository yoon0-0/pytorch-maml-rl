import maml_rl.envs
import gym
import torch
import json
import numpy as np
from tqdm import trange

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSamplerRay
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
import wandb

wandb.init(project="maml_test")

def main(args):
    wandb.config.update(args)

    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])

    if 'adapt-env-name' in config:
        print(config['adapt-env-name'])
        adapt_env = gym.make(config['adapt-env-name'], **config['env-kwargs'])
        adapt_env.close()
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    # Baseline
    if 'adapt-env-name' in config:
        baseline = LinearFeatureBaseline(get_input_size(env))
    else:
        baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    if 'adapt-env-name' in config:
        # TODO
        sampler = MultiTaskSamplerRay(config['adapt-env-name'],
                        env_kwargs=config['env-kwargs'],
                        batch_size=config['fast-batch-size'],
                        policy=policy,
                        baseline=baseline,
                        env=env,
                        seed=args.seed,
                        num_workers=config['meta-batch-size'])
        pass
    else:
        sampler = MultiTaskSamplerRay(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=config['meta-batch-size'])

    logs = {'tasks': []}
    num_iterations = 0
    train_returns, valid_returns = [], []
    for batch in trange(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        (train_episodes, valid_episodes), x_diff = sampler.sample(tasks,
                                                        num_steps=config['num-steps'],
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)

        # logs['tasks'].extend(tasks)
        logs.update(tasks=tasks,
                    num_iterations=num_iterations,
                    train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))
        # train_returns.append(get_returns(train_episodes[0]))
        # valid_returns.append(get_returns(valid_episodes))
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)

    # logs['train_returns'] = np.concatenate(train_returns, axis=0)
    # logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

    with open(args.output, 'wb') as f:
        np.savez(f, **logs)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Test')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file')
    parser.add_argument('--policy', type=str, required=True,
        help='path to the policy checkpoint')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=1,
        help='number of batches (default: 1)')
    evaluation.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch (default: 40)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output', type=str, required=True,
        help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
