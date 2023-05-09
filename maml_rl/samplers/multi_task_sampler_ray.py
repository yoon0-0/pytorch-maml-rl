import torch
import torch.multiprocessing as mp
import asyncio
import threading
import time
import numpy as np
from datetime import datetime, timezone
from copy import deepcopy

from maml_rl.samplers.sampler import Sampler, make_env
from maml_rl.envs.utils.sync_vector_env import SyncVectorEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss, get_returns
import wandb

import ray

class MultiTaskSamplerRay(Sampler):
    """Vectorized sampler to sample trajectories from multiple environements.

    Parameters
    ----------
    env_name : str
        Name of the environment. This environment should be an environment
        registered through `gym`. See `maml.envs`.

    env_kwargs : dict
        Additional keywork arguments to be added when creating the environment.

    batch_size : int
        Number of trajectories to sample from each task (ie. `fast_batch_size`).

    policy : `maml_rl.policies.Policy` instance
        The policy network for sampling. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    baseline : `maml_rl.baseline.LinearFeatureBaseline` instance
        The baseline. This baseline is an instance of `nn.Module`, with an
        additional `fit` method to fit the parameters of the model.

    env : `gym.Env` instance (optional)
        An instance of the environment given by `env_name`. This is used to
        sample tasks from. If not provided, an instance is created from `env_name`.

    seed : int (optional)
        Random seed for the different environments. Note that each task and each
        environement inside every process use different random seed derived from
        this value if provided.

    num_workers : int
        Number of processes to launch. Note that the number of processes does
        not have to be equal to the number of tasks in a batch (ie. `meta_batch_size`),
        and can scale with the amount of CPUs available instead.
    """
    def __init__(self,
                 env_name,
                 env_kwargs,
                 batch_size,
                 policy,
                 baseline,
                 env=None,
                 seed=None,
                 num_workers=1):
        super(MultiTaskSamplerRay, self).__init__(env_name,
                                               env_kwargs,
                                               batch_size,
                                               policy,
                                               seed=seed,
                                               env=env)

        self.num_workers = num_workers

        ray.init()

        self.sampler = SamplerWorker(0,
                            env_name,
                            env_kwargs,
                            batch_size,
                            self.env.observation_space,
                            self.env.action_space,
                            self.policy,
                            deepcopy(baseline),
                            num_workers,
                            self.seed)

        self._waiting_sample = False
        self._event_loop = asyncio.get_event_loop()
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def sample_tasks(self, num_tasks):
        return self.env.unwrapped.sample_tasks(num_tasks)

    def sample(self,
        tasks,
        num_steps=1,
        fast_lr=0.5,
        gamma=0.95,
        gae_lambda=1.0,
        device='cpu'):
      
        episodes = self.sampler.sample(tasks,
            num_steps,
            fast_lr,
            gamma,
            gae_lambda,
            device)

        return episodes

class SamplerWorker(object):
    def __init__(self,
                 index,
                 env_name,
                 env_kwargs,
                 batch_size,
                 observation_space,
                 action_space,
                 policy,
                 baseline,
                 num_workers,
                 seed):

        self.index = index
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.workers = [RolloutWorker.remote(
            index=index,
            batch_size=batch_size,
            env_name=env_name,
            env_kwargs=env_kwargs,
            baseline=deepcopy(baseline),
        )
        for index in range(num_workers)]
        self.seed = seed

    def sample(self,
            tasks,
            num_steps=1,
            fast_lr=0.5,
            gamma=0.95,
            gae_lambda=1.0,
            device='cpu',
            TEST=True):

            # TODO: num_step=2 이상에서 되게 하기
            params=None
            params_list=[None]*len(tasks)
            train_episodes_list = []
            adapt_rewards = []
            adapt_x_diffs = []
            for step in range(num_steps):
                train_episodes, x_diff = self.create_episodes(tasks,
                                                    params=params_list,
                                                    gamma=gamma,
                                                    gae_lambda=gae_lambda,
                                                    device=device)
                train_episodes_list.append(train_episodes)
                for i, train_episode in enumerate(train_episodes):
                    train_episode.log('_enqueueAt', datetime.now(timezone.utc))

                    loss = reinforce_loss(self.policy, train_episode, params=params_list[i])
    
                    params_list[i] = self.policy.update_params(loss,
                                                    params=params_list[i],
                                                    step_size=fast_lr,
                                                    first_order=True)
                if TEST:
                    print('TEST MODE')
                    adapt_return = np.mean(get_returns(train_episodes))
                    if True: # TEST
                        wandb.log({
                            'adapt_returns': adapt_return,
                            'x_diff': x_diff
                        })
                    adapt_rewards.append(adapt_return)
                    adapt_x_diffs.append(x_diff)
                    eval_reward_np = np.array(adapt_rewards)
                    eval_x_diff_np = np.array(adapt_x_diffs)
                    np.save('test/reward{}.npy'.format(self.seed), eval_reward_np)
                    np.save('test/x_diff{}.npy'.format(self.seed), eval_x_diff_np)


            valid_episodes, x_diff = self.create_episodes(tasks,
                                                params=params_list,
                                                gamma=gamma,
                                                gae_lambda=gae_lambda,
                                                device=device)
            # TODO: num_step=2 이상에서 되게 하기
            return (deepcopy(train_episodes_list), deepcopy(valid_episodes)), x_diff

    def create_episodes(self,
                tasks,
                params,
                gamma=0.95,
                gae_lambda=1.0,
                device="cpu"):
        episodes_ops = []
        t0 = time.time()
        for index, task in enumerate(tasks):
            episodes_ops.append(
                self.workers[index].create.remote(task=task,
                                                policy=self.policy,
                                                params=params[index],
                                                gamma=gamma,
                                                gae_lambda=gae_lambda,
                                                device=device)
            )
        episodes = ray.get(episodes_ops)
        x_diff = episodes[0][1]
        episode_list = []
        for episode in episodes:
            episode_list.append(episode[0])
        # episodes = episodes
        return episode_list, x_diff

@ray.remote
class RolloutWorker(object):
    def __init__(self,
                index,
                batch_size,
                env_name,
                env_kwargs,
                baseline,
                ) -> None:
        self.index = index
        self.baseline = baseline
        self.batch_size = batch_size
        env_kwargs['index'] = index
        self.env = make_env(env_name, env_kwargs=env_kwargs)()
        # np.random.seed(index)
        # torch.manual_seed(index)
        # self.workers = [RolloutWorker2.remote(
        #     index=index,
        #     batch_size=batch_size,
        #     env_name=env_name,
        #     env_kwargs=env_kwargs,
        #     baseline=deepcopy(baseline),
        # )
        # for index in range(self.batch_size)]

    # TODO: 하드 코딩 바꾸기
    def sample_trajectories(self, task, policy, params):
        self.env.reset_task(task)
        for i in range(self.batch_size):
            done = False
            observations = self.env.reset()
            step = 0
            with torch.no_grad():
                while step < 300 and not done: # 120 ant, halfcheetah 300 snapbot
                    self.env.render()
                    observations_tensor = torch.from_numpy(observations).type(torch.float32)
                    
                    # print('observations_tensor.shape', observations_tensor.shape)
                    if self.env.test:
                        num_zeros = 0
                        for idx in self.env.malfun_leg_idxs:
                            observations_tensor = torch.cat([observations_tensor[:num_zeros+7+4*idx], torch.zeros(4), observations_tensor[num_zeros+7+4*idx:]], dim=0) # add 4 zero
                            num_zeros += 4
                            observations_tensor = torch.cat([observations_tensor[:num_zeros+23+6+4*idx], torch.zeros(4), observations_tensor[num_zeros+23+6+4*idx:]], dim=0) # add 4 zero
                            num_zeros += 4
                            observations_tensor = torch.cat([observations_tensor[:num_zeros+23+22+7+4*idx], torch.zeros(4), observations_tensor[num_zeros+23+22+7+4*idx:]], dim=0) # add 4 zero
                            num_zeros += 4
                            observations_tensor = torch.cat([observations_tensor[:num_zeros+23+22+23+6+4*idx], torch.zeros(4), observations_tensor[num_zeros+23+22+23+6+4*idx:]], dim=0) # add 4 zero
                            num_zeros += 4
                            observations_tensor = torch.cat([observations_tensor[:num_zeros+23+22+23+22+2*idx], torch.zeros(2), observations_tensor[num_zeros+23+22+23+22+2*idx:]], dim=0) # add 4 zero
                            num_zeros += 2
                    # print('observations_tensor.shape', observations_tensor.shape)
                    pi = policy(observations_tensor, params)
                    actions_tensor = pi.sample()
                    actions = actions_tensor.cpu().numpy()
                    # print("actions.shape", actions.shape)
                    # print("self.env.malfun_leg_idxs", self.env.malfun_leg_idxs)
                    if not self.env.test:
                        for idx in self.env.malfun_leg_idxs:
                            actions[2*idx:2*idx+2] = 0
                    else:
                        # print(actions.shape)
                        deleted_idx = []
                        for idx in self.env.malfun_leg_idxs:
                            deleted_idx.append(2*idx)
                            deleted_idx.append(2*idx+1)
                        actions = np.delete(actions, deleted_idx)

                    new_observations, rewards, done, info = self.env.step(actions)
                    
                    # print("observation.shape", observations.shape)
                    if self.env.test:
                        num_zeros = 0
                        for idx in self.env.malfun_leg_idxs:
                            observations = np.insert(observations, num_zeros+7+4*idx, np.zeros(4))#torch.cat([observations[:9+4*idx], torch.zeros(4), observations[9+4*idx:]], dim=0) # add 4 zero
                            num_zeros += 4
                            observations = np.insert(observations, num_zeros+23+6+4*idx, np.zeros(4))#torch.cat([observations[:num_zeros+28+8+4*idx], torch.zeros(4), observations[4+28+8+4*idx:]], dim=0) # add 4 zero
                            num_zeros += 4
                            observations = np.insert(observations, num_zeros+23+22+7+4*idx, np.zeros(4))#torch.cat([observations[:num_zeros+28+29+9+4*idx], torch.zeros(4), observations[num_zeros+28+29+9+4*idx:]], dim=0) # add 4 zero
                            num_zeros += 4
                            observations = np.insert(observations, num_zeros+23+22+23+6+4*idx, np.zeros(4))#torch.cat([observations[:num_zeros+28+29+28+8+4*idx], torch.zeros(4), observations[num_zeros+28+29+28+8+4*idx:]], dim=0) # add 4 zero
                            num_zeros += 4
                            observations = np.insert(observations, num_zeros+23+22+23+22+2*idx, np.zeros(2))#torch.cat([observations[:num_zeros+28+29+28+28+2*idx], torch.zeros(2), observations[num_zeros+28+29+28+28+2*idx:]], dim=0) # add 4 zero
                            num_zeros += 2
                            actions = np.insert(actions, 2*idx, np.zeros(2))
                        

                    yield (observations, actions, rewards, i)
                    step+=1
                    observations = new_observations

    def create(self,
                task,
                policy,
                params=None,
                gamma=0.95,
                gae_lambda=1.0,
                device="cpu"):
        episodes = BatchEpisodes(batch_size=self.batch_size,
                    gamma=gamma,
                    device=device)
        episodes.log('_createdAt', datetime.now(timezone.utc))

        observations_list=[]
        actions_list=[]
        rewards_list=[]
        batch_ids_list=[]
        t0 = time.time()

        # episodes_ops = []
        # for worker in self.workers:
        #     episodes_ops.append(
        #         worker.get_episodes.remote(task, policy, params)
        #     )
        # episodes_list = ray.get(episodes_ops)

        # for episode in episodes_list:
        #     observations, actions, rewards, batch_ids = episode
        #     observations_list = observations_list + observations
        #     actions_list = actions_list + actions
        #     rewards_list = rewards_list + rewards
        #     batch_ids_list = batch_ids_list + batch_ids
        # TODO: logging info
        for observations, actions, rewards, batch_ids in self.sample_trajectories(task, policy, params):
            observations_list.append(observations)
            actions_list.append(actions)
            rewards_list.append(rewards)
            batch_ids_list.append(batch_ids)
        episodes.append(observations_list, actions_list, rewards_list, batch_ids_list)
        episodes.log('duration', time.time() - t0)
        self.baseline.fit(episodes)
        episodes.compute_advantages(self.baseline,
                                    gae_lambda=gae_lambda,
                                    normalize=True)
        return [episodes, self.env.sim.data.qpos[0]]

# @ray.remote
# class RolloutWorker2(object):
#     def __init__(self,
#                 index,
#                 batch_size,
#                 env_name,
#                 env_kwargs,
#                 baseline,
#                 ) -> None:
#         self.index = index
#         self.baseline = baseline
#         self.batch_size = batch_size
#         env_kwargs['index'] = index
#         self.env = make_env(env_name, env_kwargs=env_kwargs)()
#         np.random.seed(index)
#         torch.manual_seed(index)
    
#     def get_episodes(self, task, policy, params):
#         observations_list=[]
#         actions_list=[]
#         rewards_list=[]
#         batch_ids_list=[]

#         for observations, actions, rewards, batch_ids in self.sample_trajectories(task, policy, params):
#             observations_list.append(observations)
#             actions_list.append(actions)
#             rewards_list.append(rewards)
#             batch_ids_list.append(batch_ids)
#         return observations_list, actions_list, rewards_list, batch_ids_list

#     def sample_trajectories(self, task, policy, params):
#         self.env.reset_task(task)
#         step=0
#         observations = self.env.reset()
#         with torch.no_grad():
#             while step < 120:
#                 # self.env.render()
#                 observations_tensor = torch.from_numpy(observations).type(torch.float32)
#                 pi = policy(observations_tensor, params)
#                 actions_tensor = pi.sample()
#                 actions = actions_tensor.cpu().numpy()

#                 new_observations, rewards, done, _ = self.env.step(actions)
#                 yield (observations, actions, rewards, self.index)
#                 step+=1
#                 observations = new_observations
