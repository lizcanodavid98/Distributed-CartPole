# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 09:04:41 2021

@author: dasan
"""
import os
import numpy as np
import json

def add_new_experiment(filename, experiment):
        with open(filename, 'r+') as f:
            if not f.read(1):
                f.write('{"experiments":[\n')
                
            else:
                f.read()
                f.seek(f.tell() - 2, os.SEEK_SET)
                f.write(',\n')
                
            f.write(experiment)
            f.write(']}')
        
def store_to_file(filename, score_eval, avg_delay, timeout, total_actions, timeouts):
    with open(filename, 'a+'): #CREA EL FICHERO SI NO EXISTE
        pass
    experiment = {
                'Execution': score_eval.execution_type,
                'Average Delay': avg_delay,
                # 'Variance': var_delay,
                'Timeout': timeout,
                'Total Actions': total_actions,
                'Total Timeouts': timeouts,
                'Mean': np.mean([row[1] for row in score_eval.score_table]),
                'Results': score_eval.score_table
            }
    
    add_new_experiment(filename, json.dumps(experiment))

def store_actions_to_file(filename, actions, episode, delay, timeout):
    with open(filename, 'a+'):
        pass
    experiment = {
        'Delay': delay,
        'Timeout': timeout,
        'Episode': episode,
        'actions': actions
    }

    add_new_experiment(filename, json.dumps(experiment))
    
def clear_file(filename):
    with open(filename, 'w'):
        pass