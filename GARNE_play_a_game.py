# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:49:22 2019

@author: Z0014354

##### TEST GARNE #####
"""

import torch
import numpy as np
import gym
import argparse
from GA_Base import base




def main(env, model, n_runs, render):
    for i in range(n_runs):
        state = env.reset()
        hidden_out = (torch.zeros((1, 1, model.hidden_size), dtype=torch.float),torch.zeros((1, 1, model.hidden_size), dtype=torch.float))
        while True:
            hidden_in = hidden_out
            state = torch.from_numpy(state).unsqueeze(0).float()
            action, hidden_out = model(state, hidden_in)
            action = action.detach().numpy()
            action = np.clip(action*env.action_space.high[0], env.action_space.low[0], env.action_space.high[0])
            next_state, reward, done, _ = env.step(action[0])
            state = next_state
            if done:
                break
    env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-env", "--environment", type=str, help="Name of the environemnt to test the trained agent")
    parser.add_argument("-model", "--saved_model", type=str, help="Name of the saved model state dict.")
    parser.add_argument("-ls", "--layer_size", type=int, default=10, help="The size of the neural network layer size. Default is: 10")
    parser.add_argument("-network", type=str, choices=["ff", "cnn", "lstm"], default="ff", help="Type of the neural network. User can choose between feed forward (ff), one dimensional convolutional network (cnn) and long-short-term-memory (lstm) network. Default is ff.")
    parser.add_argument("-render", type=bool, default=False, help="Renders the environment")
    parser.add_argument("-runs", type=int, default=1, help="Number of validation runs, default is 1!")
    parser.add_argument("-data", "--validation_data", type=str, default="no_data", help="For the Damper with real street data it is possible to load different real road segments: data is the name of that file which consists of the street points")
    args = parser.parse_args()
    
    
    env_name = args.environment
    state_dict_name = args.saved_model
    rendering = args.render
    number_of_runs = args.runs
    model_type = args.network
    hidden_size = args.layer_size
    
    env = gym.make(env_name)
    
    if args.validation_data != "no_data":
        assert args.environment == "DamperReal-v0" or args.environment == "DamperReal-v1", "Wrong environment selected. Street data can only loaded for DamperReal-v0 or DamperReal-v1 environment!"
        env.unwrapped.load_data(args.validation_data)
    
    
    action_type = type(env.action_space)
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    
    #adding high low values...
    
   
    #building the model:
    network = base.build_net(env=env, seeds=[0], model_type=model_type, hidden_size=hidden_size, noise_std=0.1, action_type=action_type)

    # loading old state_dict:
    network.load_state_dict(torch.load(state_dict_name))
    
    # test run:    
    main(env, network, number_of_runs, rendering)
    

    






