# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:24:06 2020

@author: dasan
"""


import aiohttp, asyncio, nest_asyncio, async_timeout
import json
import gym
from collections import deque
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
import random

nest_asyncio.apply()
session = aiohttp.ClientSession()

url = 'http://127.0.0.1:8080'




async def fetch_post(session, url, data_json):
    async with session.post(url, data = data_json) as resp:
        pass
        # print(url + ' ' + str(resp.status))
        # print(await resp.text())
        
async def post_json(data_json, url=url):
    await fetch_post(session, url, data_json)
          
async def fetch_post_json(session, url, data_json):
    async with session.post(url, data = data_json) as resp:
        return await resp.json()

async def post_expected_json(data_json, url = url):
    data = await fetch_post_json(session, url, data_json)
    # print("Action: {}\nState: {}".format(data['action'], data['state']))
    return data

async def get_something(url = url):
    await fetch_get(session, url)
    
async def fetch_get(session, url):
    async with session.get(url) as resp:
        pass
        # print(url + ' ' + str(resp.status))
        # print(await resp.text())
    
async def close_session():
    await session.close()
   
    
    
    
class TrainSolver:

    def __init__(self, max_episodes):
        self.max_episodes = max_episodes
        self.score_table = deque(maxlen=400)
        self.average_of_last_runs = None
        self.play_episodes = 100
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n 
    
    async def train(self):
        env = gym.make('CartPole-v1')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        episode = 0
        
        data = {
        'action': 0,
        'state': [0,0,0,0],
        'state_next': [0,0,0,0],
        'reward': 1.0,
        'done': True,
        'info': {}
        }
        
        solver_params = json.dumps({
            'observation-space': observation_space, 
            'action-space': action_space
            })
    
        solver_url = url + '/solver'
        await post_json(solver_params, url = solver_url)
        
        # score_eval = ScoreEvaluator(400, 50, self.model)
        score_params = json.dumps({
            'max-len':400,
            'average_of_last_runs': 50
            })
        
        score_url = url + '/score_evaluator'
        await post_json(score_params, url = score_url)
        
        while episode < self.max_episodes:
            episode += 1
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            step = 0
            while True:
                
                if type(state) != list:
                    data["state"] = state.tolist()
                else:
                    data["state"] = state
        
                data_json = json.dumps(data)
                
                
                step += 1
                # env.render()
                
                # action = self.solver.take_action(state)
                try:
                    async with async_timeout.timeout(0.2):
                        data = await post_expected_json(data_json)
                        action = data["action"]
                except asyncio.TimeoutError:
                    # print("No response")
                    action = data["action"]
                    
                state_next, reward, done, info = env.step(action)
                if not done:
                    reward = reward
                else:
                    reward = -reward
                state_next = np.reshape(state_next, [1, observation_space])
                
                # self.solver.add_to_memory(state, action, reward, state_next, done)
                data["state_next"] = state_next.tolist()
                data["reward"] = reward
                data["done"] = done
                data["info"] = info
                data["action"] = action
                data_json = json.dumps(data)
                memory_url = url + '/memory'
                await post_json(data_json, url = memory_url)
                
                state = state_next

                if done:
                    print("Run: " + str(episode) + ", score: " + str(step))
                    # self.score_table.append([episode, step])
                    
                    # score_eval.store_score(episode, step)
                    score_json = json.dumps({
                        'episode': episode,
                        'step': step
                        })
                    score_url = url + '/score'
                    await post_json(score_json, url = score_url)
                    break
                
                # self.solver.experience_replay()
                replay_url = url + '/replay'
                await get_something(url = replay_url)
         
        # score_eval.plot_evaluation("Training")    
        plot_url = url + '/plot'
        await post_json(json.dumps({'text':'Training'}), url = plot_url)
        
    
# async def train():
#     env = gym.make('CartPole-v1')
#     observation_space = env.observation_space.shape[0]
#     action_space = env.action_space.n
    
    
#     solver_params = json.dumps({
#         'observation-space': observation_space, 
#         'action-space': action_space
#         })
    
#     solver_url = url + '/solver'
#     await post_json(solver_params, url = solver_url)
    
#     score_params = json.dumps({
#         'max-len':400,
#         'average_of_last_runs': 50
#         })
#     score_url = url + '/score_evaluator'
#     await post_json(score_params, url = score_url)
    
    

#     data = {
#         'action': 0,
#         'state': [0,0,0,0],
#         'state_next': [0,0,0,0],
#         'reward': 1.0,
#         'done': True,
#         'info': {}
#         }
    
    
#     i = 0
#     while i != 1:
#         i = i+1
#         if data["done"] == True:
#             state = env.reset()
    
#         if type(state) != list:
#             data["state"] = state.tolist()
#         else:
#             data["state"] = state
        
#         data_json = json.dumps(data)
        
#         try:
#             async with async_timeout.timeout(0.2):
#                 data = await post_expected_json(data_json)
#                 action = data["action"]
#         except asyncio.TimeoutError:
#             # print("No response")
#             action = 0
        
        
        
#         state_next, reward, done, info = env.step(action)
#         data["state_next"] = state_next.tolist()
#         data["reward"] = reward
#         data["done"] = done
#         data["info"] = info
#         data["action"] = action
        
#         data_json = json.dumps(data)
#         # print("Action: {}\nState: {}\nNext state: {}\nReward: {}\nDone: {}\nInfo: {}".format(data["action"], data["state"], data["state_next"], data["reward"], data["done"], data["info"]))
#         print("Action: {}".format(data["action"]))
        
        
#         state = data["state_next"]
#         data_json = json.dumps(data)
#         memory_url = url + '/memory'
#         await post_json(data_json, url = memory_url)
        
#         score_json = json.dumps({
#             'episode': episode,
#             'step': step
#             })
#         score_url = url + '/score'
#         await post_json(score_json, url = score_url)

if __name__ == '__main__':
    trainer = TrainSolver(5)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(trainer.train())
    loop.run_until_complete(close_session())
  
