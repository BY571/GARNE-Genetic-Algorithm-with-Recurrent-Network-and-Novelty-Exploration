# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:30:48 2019


@author: Z0014354
"""



import gym
import numpy as np
from GA_Addons import crossover, novelty
from GA_Base import base
import multiprocessing as mp
from tensorboardX import SummaryWriter
from collections import deque, namedtuple
import argparse
import time
import matplotlib.pyplot as plt
import torch

HIDDEN_SIZE = 10
K_NEIGHBORS = 25
RENDER_EVERY = 10 
OutputItem = namedtuple('OutputItem', field_names=["seeds", "reward", "steps", "bc"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-info", type=str, help="Information or name of the run")   
    parser.add_argument("-env", type=str, default="Pendulum-v0", help="The environment you want to train the algorithm. Default is Pendulum-v0")
    parser.add_argument("-ps", "--Population_size", type=int, default=500, help="Population size. Default is: 500")
    parser.add_argument("-pc", "--Parent_count", type=int, default=10, help="Number of top performer of the population that build the new population. Default is: 10")
    parser.add_argument("-g", "--Generation_max", type=int, default=200, help="Maximum number of generations. Default is: 200")
    parser.add_argument("-ls", "--layer_size", type=int, default=64, help="The size of the neural network layer size. Default is: 64")
    parser.add_argument("-network", type=str, choices=["ff", "cnn", "lstm"], default="ff", help="Type of the neural network. User can choose between feed forward (ff), one dimensional convolutional network (cnn) and long-short-term-memory (lstm) network. Default is ff.")
    parser.add_argument("-std", "--mutation_std", type=float, default=0.1, help="The noise that is added to the network weights as a mutation. Default is 0.1")
    parser.add_argument("-novelty", type=bool, default=False, help="Adds novelty search to the algorithm. Default is: False \n If choosen, be aware to adapt K_NEIGHBORS value and the behavior characterization (bc in GA_Addon/base.py) depending on the task!")
    parser.add_argument("-crossover", type=int, choices=range(0, 3), default=0, help="Adds crossover to the creation process of the new population. The user has two crossover methods to choose from: \nMethod 1:  Slices the seeds of both parents in the middle and combines them. keyword argument: 1 \nMethod 2: For each seed in seed_index of both parents. Pick one for the child with a 50/50 prob. keyword argument 2. Default is 0 - No crossover!")
    parser.add_argument("--worker_count", type=int, default=10, help="Numbers of worker that gather training data")
    parser.add_argument("-save_model", type=str, default="no_saving", help="Saving the best model after training in the current directory")
    parser.add_argument("--save_every", type=int, default=10, help="Saving the best Performer after X generations, default is 10")
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Parse Arguments
    env_name = args.env
    POPULATION_SIZE = args.Population_size
    PARENTS_COUNT = args.Parent_count
    max_generation = args.Generation_max
    HIDDEN_SIZE = args.layer_size
    model_type = args.network
    NOISE_STD = args.mutation_std
    NOVELTY_USE = args.novelty 
    CROSSOVER_METHOD = args.crossover
    WORKERS_COUNT = args.worker_count
    save_every = args.save_every
    
    K_NEIGHBORS = 25 # for novelty calculation
    SEEDS_PER_WORKER = POPULATION_SIZE // WORKERS_COUNT
    MAX_SEED = 2**31
    ARCHIVE_PROB = 0.01   # Probability that a behavior characterization gets added to the archive
    
    
    env = gym.make(env_name)
    action_type = type(env.action_space)
    if args.info:
        writer = SummaryWriter("runs/"+args.info)
    else: 
        writer = SummaryWriter()
    # create to store behavior characterization
    archive = []
    reward_buffer = deque(maxlen=15)  # buffer to store the reward gradients to see if rewards stay constant over a defined time horizont ~> local min
    W = 1
    
    queues_in = []
    queue_out = mp.Queue(maxsize=WORKERS_COUNT)
    workers = []
    
    mean_rewards = []
    max_rewards = []
    min_rewards =[]
    std_rewards = []
    
    for _ in range(WORKERS_COUNT):
        # starts worker processes and sets init seed in worker queue
        queue_in = mp.Queue(maxsize=1)
        queues_in.append(queue_in)
        worker = mp.Process(target=base.worker_func, args=(queue_in, queue_out, model_type, HIDDEN_SIZE, NOVELTY_USE, env_name, NOISE_STD, action_type))
        worker.start()
        seeds = [(np.random.randint(MAX_SEED),) for _ in range(SEEDS_PER_WORKER)]
        queue_in.put(seeds)
    
    print("All running!")
    
    overall_best_performer = None
    overall_best_score = - np.inf


    elite = None
    overall_steps = 0
    for gen in range(1,max_generation+1):
        #batch_step = 0
        population = []
        bc_storage = []
        S = np.minimum(K_NEIGHBORS, len(archive))
        # collect all seeds and rewards of the worker
        while len(population) < SEEDS_PER_WORKER * WORKERS_COUNT:
            out_item = queue_out.get()
            reward = out_item.reward
            bc = out_item.bc
            bc_storage.append(bc)
            
            if NOVELTY_USE and len(archive) > 0:
                # calcs the novelty for each pop member
                distance = novelty.get_kNN(archive=archive, bc=bc, n_neighbors=S)            
                novelty_ = distance / S
                # calc new reward _weighted reward_novelty
                reward = (W*reward) + ((1-W)*novelty_) 

            
            # build population 
            population.append((out_item.seeds, reward))
            #batch_step += out_item.steps
            overall_steps += out_item.steps
        if elite is not None:
            population.append(elite)
        
        # append bcs to archive
        if NOVELTY_USE:
            archive = novelty.add_bc_to_archive(bc_storage=bc_storage, archive=archive, archive_prob=ARCHIVE_PROB)


        # sort population based on their reward max->min
        population.sort(key=lambda p: p[1], reverse=True)
        # take the top X rewards 
        reward = [p[1] for p in population[:PARENTS_COUNT]]
        # set top performer as new elite
        elite = population[0]
        
        #save overall best performer
        if elite[1] > overall_best_score:
            overall_best_performer = elite[0]
        
        
        # monitoring 
        reward_mean = np.mean(reward)

        writer.add_scalar("mean_reward", reward_mean, global_step=gen)
        writer.add_scalar("max_reward", np.max(reward_mean), global_step=gen)
        writer.add_scalar("min_reward", np.min(reward_mean), global_step=gen)
        writer.add_scalar("std_reward", np.std(reward_mean), global_step=gen)
        
        

        if NOVELTY_USE:
            if len(reward_buffer) > 0: 
                reward_gradient_mean = np.mean(reward_buffer)
            else:
                reward_gradient_mean = 0
            r_koeff = abs(reward_mean - reward_gradient_mean)
            # if last few rewards are almost konstant  -> stuck in loc minima -> decrease W for exploration: higher novelty weight
            if r_koeff < np.std(reward_buffer):
                W = np.maximum(0, W - 0.025)
            else:
                W = np.minimum(1, W + 0.025)
            reward_buffer.append(reward_mean)
            writer.add_scalar("reward_weight_koeff", W, global_step=gen)
            
        # loop over each worker get parrent
        for worker_queue in queues_in:
            seeds = []
            # each worker has several networks to run -> several seeds
            for _ in range(SEEDS_PER_WORKER):
                # sample parent seed
                parent1_idx = np.random.randint(PARENTS_COUNT)
                parent2_idx = np.random.randint(PARENTS_COUNT)
                parent1 = list(population[parent1_idx][0])
                parent2 = list(population[parent2_idx][0])

                if CROSSOVER_METHOD == 1:
                    parent_seeds = crossover.slice_parents(parent1, parent2)
                elif CROSSOVER_METHOD == 2:
                    parent_seeds = crossover.pick_random(parent1, parent2)
                else:
                    # no crossover
                    parent_seeds = parent1
                # sample new seed
                next_seed = np.random.randint(MAX_SEED)
                # create new population 
                seeds.append(tuple(parent_seeds + [next_seed]))
                
            worker_queue.put(seeds)

        print("\rGeneration: {} | Steps: {} mio | Mean_Reward: {:.2f}  ".format(gen, overall_steps/1e6, reward_mean), end = "", flush = True)
        if gen % save_every == 0:
            seeds = elite[0]
            network = base.build_net(env=env, seeds=seeds, model_type=model_type, hidden_size=HIDDEN_SIZE, noise_std=NOISE_STD, action_type=action_type)
            torch.save(network.state_dict(), "Gen"+str(gen)+".pt")
            
        
    for worker in workers:
        worker.join()
    
    
    ## save the last elite network as output agent:
    if args.save_model != "no_saving":
        
        network = base.build_net(env=env, seeds=overall_best_performer, model_type=model_type, hidden_size=HIDDEN_SIZE, noise_std=NOISE_STD, action_type=action_type)
        torch.save(network.state_dict(), args.save_model+".pt")
        
        
    env.close()
    end_time = time.time()
    print("Training took: {} min".format((end_time-start_time)/60))

    