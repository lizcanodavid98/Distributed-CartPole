# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:24:06 2020

@author: dasan
"""


import json
import gym
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
import random
import os


from connectivity import get_action, load_model, start_solver, store_in_memory, replay_experience, new_session
from logs import store_to_file, store_actions_to_file, clear_file
from probe import get_timestamp

class ScoreEvaluator:

    def __init__(self, max_len, average_of_last_runs, execution_type):
        self.max_len = max_len
        self.score_table = []
        self.average_of_last_runs = average_of_last_runs
        self.execution_type = execution_type

    def store_score(self, episode, step, client_actions, server_actions):
        self.score_table.append([episode, step, client_actions, server_actions])

    def plot_evaluation(self, title="Training"):
        # avg_score = mean(self.score_table[1])
        x = []
        y = []
        for i in range(len(self.score_table)):
            x.append(self.score_table[i][0])
            y.append(self.score_table[i][1])

        average_range = self.average_of_last_runs if self.average_of_last_runs is not None else len(x)
        plt.plot(x, y, label="score per run")
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--",
                 label="last " + str(average_range) + " runs average")
        title = "CartPole-v1 " + str(title)
        plt.title(title)
        plt.xlabel("Runs")
        plt.ylabel("Score")
        plt.show()
    
    
class TrainSolver:

    def __init__(self, max_episodes):
        self.max_episodes = max_episodes
        self.average_of_last_runs = None
        self.play_episodes = 100
        self.delay = 0
        self.timeout = 60
    
          
    def play(self, play_episodes = 100, model = None):
        self.play_episodes = play_episodes
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        episode = 0
        total_actions = 0
        
        start_solver(session, observation_space, action_space)

        score_eval = ScoreEvaluator(400, 50, 'Play')
        
        if model is None:
            print("Can't load specified model.")
        else:
            load_model(session, model)
        
        
        data = {
            'action': 0,
            'state': [0,0,0,0],
            'state_next': [0,0,0,0],
            'reward': 1.0,
            'done': True,
            'info': {}
        }
        
        while episode < self.play_episodes:
            episode += 1
            state = env.reset()
            pending_tasks = []
            actions = []
            session.client_actions = [0,0,0]
            session.server_actions = 0
            
            if type(state) != list:
                data["state"] = state.tolist()
            else:
                data["state"] = state
            
            step = 0
            
            while True:
                step += 1
                # env.render()
                action = data['action']
                total_actions += 1
                # print('Before Get Action: ' + get_timestamp())
                data = get_action(session, data, pending_tasks, delay = self.delay, timeout = self.timeout)
                # print('After Get Action: ' + get_timestamp())

                # print(session.timeouts)
                actions.append(data['action'])
                
                if data['action'] == 2:
                    session.client_actions[0] += 1
                    data['action'] = action
                elif data['action'] == 3:
                    session.client_actions[1] += 1
                    data['action'] = action
                elif data['action'] == 4:
                    session.client_actions[2] += 1
                    data['action'] = action
                else: 
                    session.server_actions += 1
                
                
                state_next, reward, done, info = env.step(data['action'])
                if not done:
                    reward = reward
                else:
                    reward = -reward
                
                data["state_next"] = state_next.tolist()
                data["reward"] = reward
                data["done"] = done
                data["info"] = info
                data["action"] = action
                
                state = state_next
                data['state'] = state.tolist()
                
                if done:
                    print('Run: ' + str(episode) + ', score: ' + str(step))
                    store_actions_to_file('actions.txt', actions, episode, self.delay, self.timeout)
                    score_eval.store_score(episode, step, session.client_actions, session.server_actions)
                    break

        score_eval.plot_evaluation(title = "Playing")
        store_to_file('results.txt', score_eval, self.delay, self.timeout, total_actions, session.timeouts)
                
    def train(self):
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        score_eval = ScoreEvaluator(400, 50, 'Train')
        
        episode = 0
        total_actions = 0
        start_solver(session, observation_space, action_space)
        
        while episode < self.max_episodes:
            episode += 1
            state = env.reset()
            pending_tasks = []
            session.client_actions = [0,0,0]
            session.server_acitons = 0
            
        
            data = {
            'action': 0,
            'state': [0,0,0,0],
            'state_next': [0,0,0,0],
            'reward': 1.0,
            'done': True,
            'info': {}
            }
            
            if type(state) != list:
                data["state"] = state.tolist()
            else:
                data["state"] = state
                
            step = 0
            
            while True:
    
                step += 1
                # env.render()
                action = data['action']
                total_actions += 1
                data = get_action(session, data, pending_tasks, training = True, delay = self.delay, timeout = self.timeout)

                if data['action'] == 2:
                    session.client_actions[0] += 1
                    data['action'] = action
                elif data['action'] == 3:
                    session.client_actions[1] += 1
                    data['action'] = action
                elif data['action'] == 4:
                    session.client_actions[2] += 1
                    data['action'] = action
                else: 
                    session.server_actions += 1

                state_next, reward, done, info = env.step(data['action'])
                if not done:
                    reward = reward
                else:
                    reward = -reward
                    
                data["state_next"] = state_next.tolist()
                data["reward"] = reward
                data["done"] = done
                data["info"] = info
                # data["action"] = action
                
                store_in_memory(session, data)
                
                state = state_next
                data['state'] = state.tolist()

                if done:
                    print("Run: " + str(episode) + ", score: " + str(step))
                    # self.score_table.append([episode, step])
                    score_eval.store_score(episode, step, session.client_actions, session.server_actions)
                    break
                replay_experience(session)

        score_eval.plot_evaluation(title = "Playing")
        store_to_file('results.txt', score_eval, self.delay, self.timeout, total_actions, session.timeouts)
        
        
        
        
if __name__ == '__main__':
    
    session = new_session(url = 'http://127.0.0.1:5000')
    clear_file('actions.txt')
    clear_file('results.txt')
    trainer = TrainSolver(150)
  
    # trainer.train()
    # for i in range(1,5):
    #     for j in range(int(i/5*10), int(i*10), 2*i):
    #         trainer.delay = j/100
    #         trainer.timeout = i/10
    #         print('Delay: {}, Timeout: {}'.format(trainer.delay, trainer.timeout))
    #         trainer.play(play_episodes = 2, model = 'cartpole_model_v3.h5')
    #         session.timeouts = 0
            
    
    for i in [0, 15, 22, 50, 100, 150, 200, 250, 300, 350, 400]:
        trainer.delay = i/1000
        trainer.timeout = 0.05
        trainer.play(play_episodes = 100, model = 'cartpole_model_v3.h5')
        session.timeouts = 0
    
    # trainer.delay = 0.1
    # trainer.timeout = 0.05
    # trainer.play(play_episodes = 2, model = 'cartpole_model_v3.h5')
    # trainer.play(play_episodes=2)