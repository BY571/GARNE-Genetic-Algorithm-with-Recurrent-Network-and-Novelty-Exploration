import gym
from GA_Base import model 
import torch
import torch.nn as nn
import copy
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt

OutputItem = namedtuple('OutputItem', field_names=["seeds", "reward", "steps", "bc"])

def evaluate(env, net):
    """
    Runs an evaluation on the given network.
    returns the summed rewards collected and the step count
    """
    state = env.reset()
    hidden_out = (torch.zeros((1, 1, net.hidden_size), dtype=torch.float),torch.zeros((1, 1, net.hidden_size), dtype=torch.float))
    reward_sum = 0
    steps_count = 0
    while True:
        hidden_in = hidden_out
        state = torch.from_numpy(state).unsqueeze(0).float()
        action, hidden_out = net(state, hidden_in)
        action = action.detach().numpy()
        action = np.clip(action*env.action_space.high[0], env.action_space.low[0], env.action_space.high[0])

        next_state, reward, done, _ = env.step(action[0])
        reward_sum += reward
        steps_count  += 1
        state = next_state
        if done:
            break
    bc = np.array([reward_sum]) 
    return reward_sum, steps_count, bc


def mutate(net, seed, noise_std, copy_net=True):
    """
    Mutates the given network parameters. Based on the applied seed a normal distributed noise is added to the network parameters    
    """
    # copy current net
    mutated_net = copy.deepcopy(net) if copy_net else net
    # set seed for mutation
    np.random.seed(seed)
    for param in mutated_net.parameters():
        noise = torch.tensor(np.random.normal(size=param.shape).astype(np.float32))
        param.data += noise * noise_std
    
    return mutated_net
        

def build_net(env, seeds, model_type, hidden_size, noise_std, action_type):
    """
    Build a network based on the seeds 
    seed[0]   ->  initial network seed
    seed[1:]  ->  mutation seeds
    """
    torch.manual_seed(seeds[0])
    if action_type == gym.spaces.box.Box:
    	action_space = env.action_space.shape[0] 
    else:
        action_space = env.action_space.n
    if model_type == "ff":
        net = model.Model_FF(env.observation_space.shape[0], action_space, hidden_size, action_type)
    elif model_type == "cnn":
        net = model.Model_CNN1D(env.observation_space.shape[0], action_space, hidden_size)
    else:
        net = model.Model_LSTM(env.observation_space.shape[0], action_space, hidden_size, action_type)
    for seed in seeds[1:]:
        net = mutate(net, seed, noise_std, copy_net=False)

    return net



def worker_func(queue_in, queue_out, model_type, hidden_size, novelty_use, env_name, noise_std, action_type):
    """
    Each worker has several init seeds -> runs more than one network for training.
    For each seed the worker runs a test run with the mutations and puts the seeds, rewards and steps in the output queue
    Also the worker saves current build networks based on a seed history. saves computation if later seeds from parent are already in the cache -> no need to rebuild the net
    
    """
    env = gym.make(env_name)
    
    cache = {} # to store population / networks
    
    while True:
        parents_seeds = queue_in.get()
        if parents_seeds == None:
            break
        new_cache = {}
        # for each network seeds 
        for seeds in parents_seeds:
            # if seed history exist
            if len(seeds) > 1:
                net = cache.get(seeds[:-1])#
                # check if network already exists
                if net is not None:
                    # if exist mutate on the new given seed -> the last in the list
                    net = mutate(net, seeds[-1], noise_std)
                else:
                    # if not exist build the net with the seed history
                    net = build_net(env, seeds, model_type, hidden_size, noise_std, action_type)
            else:
                # since no seed history exist -> build network
                net = build_net(env, seeds, model_type, hidden_size, noise_std, action_type)
            
            # saves the networks in a cache 
            new_cache[seeds] = net
            # evaluate new network mutation
            reward, steps, bc = evaluate(env, net)
            queue_out.put(OutputItem(seeds=seeds, reward=reward, steps=steps, bc=bc))
        # after evaluating all seeds the worker sets the new_cache with saved nets to the current cache
        cache = new_cache
    

def test_run(env, model_type, hidden_size,action_type, seeds, noise_std, render = False):
    """
    Runs a test run for evaluation and monitoring the outputs
    """

    net = build_net(env, seeds, model_type, hidden_size, noise_std,action_type)
    hidden_out = (torch.zeros((1, 1, hidden_size), dtype=torch.float),torch.zeros((1, 1, hidden_size), dtype=torch.float))
    state = env.reset()
 
    for step in range(2000):
        hidden_in = hidden_out
        if step % 5 == 0:
            if render:
                env.render()
        
        state = torch.from_numpy(state).float()
        action, hidden_out = net(state, hidden_in)
        action = action.detach().numpy()
        action = np.clip(action*env.action_space.high[0], env.action_space.low[0], env.action_space.high[0])

        state, reward, done, info = env.step(action[0])
        if done:
            plt.close()
            break
