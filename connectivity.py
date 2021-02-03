import asyncio
import nest_asyncio
import json
import requests

class mySession:
    def __init__(self, url):
        self.url = url
        nest_asyncio.apply()
        self.timeouts = 0
    
def new_session(url):
    s = mySession(url)
    return s

def start_solver(session, observation_space, action_space):
    solver_params = json.dumps({
            'observation-space': observation_space, 
            'action-space': action_space
            })
    
    solver_url = session.url + '/solver'
    requests.post(solver_url, data = solver_params)


def load_model(session, model):
    load_model_url = session.url + '/load'
    requests.post(load_model_url, data = json.dumps({'text': model}))
    

def get_action(session, data, pending_tasks, delay = 0, timeout = 60, training = False):
    
    data = asyncio.run(__get_action(session, data, training, pending_tasks, delay, timeout))
    return data

async def __get_action(session, data, training, pending_tasks, delay, timeout):
    if training:
        play_url = session.url
    else:
        play_url = session.url + '/play'
    task = asyncio.create_task(__get_data(data, play_url, delay))

    try:
        result = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        return result
    except asyncio.TimeoutError:
        pending_tasks.append(task)
        # print('Timeout')
        session.timeouts += 1

    if len(pending_tasks) > 0:
        pending = pending_tasks[len(pending_tasks)-1]

        if pending.done():
            pending_tasks.pop()
            return pending.result()
        else:
            return find_last_done(len(pending_tasks)-2, pending_tasks, data)
    else:
        data['action'] = 2
    
    return data


async def __get_data(data, url, delay):
    r = requests.post(url, data = json.dumps(data))

    data = json.loads(r.text)
    
    await asyncio.sleep(delay)
    return data

def find_last_done(index, pending_tasks, data):
    if index > -1:
        pending = pending_tasks[index]
        if pending.done():
            for i in range(index+1):
                pending_tasks.pop(i)
            return pending.result()
        else:
            return find_last_done(index-1, pending_tasks, data)

    else:
        data['action'] = 2
        return data

def store_in_memory(session, data):
    memory_url = session.url + '/memory'
    requests.post(memory_url, data = json.dumps(data))

def replay_experience(session):
    replay_url = session.url + '/replay'
    requests.get(replay_url)