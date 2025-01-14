import gym
import torch
import json
import os
import yaml
from tqdm import trange
import numpy as np
import maml_rl.envs
from maml_rl.metalearners import MAMLTRPORAY
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSamplerRay
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
import wandb
wandb_flag = True
if wandb_flag:
    wandb.init(project="maml")

def main(args):
    if wandb_flag:
        wandb.config.update(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')
        reward_filename = os.path.join(args.output_folder)

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    policy.share_memory()
    if wandb_flag:
        wandb.watch(policy)

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSamplerRay(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=config['meta-batch-size'])#args.num_workers)

    metalearner = MAMLTRPORAY(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)

    num_iterations = 0
    eval_rewards = []
    for batch in trange(config['num-batches']):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        episodes, x_diff = sampler.sample(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        logs = metalearner.step(*episodes,
                                # num_tasks=config['fast-batch-size'],
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = episodes
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(tasks=tasks,
                    num_iterations=num_iterations,
                    train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))
        if wandb_flag:
            wandb.log({
                'train_returns': np.mean(logs['train_returns']),
                'valid_returns': np.mean(logs['valid_returns']),
                'loss': np.mean(logs['loss_after']),
                'kl_after': np.mean(logs['kl_after']),
                'x_diff': x_diff
            })
        eval_rewards.append(np.mean(logs['valid_returns']))

        # Save policy
        if args.output_folder is not None:
            with open(policy_filename+str(batch), 'wb') as f:
                torch.save(policy.state_dict(), f)

        eval_reward_np = np.array(eval_rewards)
        np.save(reward_filename+'/reward{}.npy'.format(args.seed), eval_reward_np)


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', default='configs/maml/snapbot-6leg-ray.yaml', type=str,# required=True,
        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', default='snapbot-6leg-ray-2', type=str,
        help='name of the output folder')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed')
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
    if wandb_flag:
        wandb.alert('finished', 'Hooray!!')
