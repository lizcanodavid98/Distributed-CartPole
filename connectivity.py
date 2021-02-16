import asyncio, aiohttp
import nest_asyncio
import json
import requests

from probe import get_microsecond, get_timestamp
import datetime

class mySession:
    def __init__(self, url):
        self.url = url
        nest_asyncio.apply()
        self.timeouts = 0
        self.server_actions = 0
        self.client_actions = [0,0,0]
        self.latency = []
        self.aiohttp_session = aiohttp.ClientSession()

    async def post_expected_json(self, data_json, url):
        data = await fetch_post_json(self.aiohttp_session, url, data_json)
        return data

    async def post_json(self, data_json, url, timeout):
        await fetch_post(self.aiohttp_session, url, data_json, timeout)

    async def get_something(self, url, timeout):
        await fetch_get(self.aiohttp_session, url, timeout)

    def close_session(self):
        asyncio.run(self.__close_session())

    async def __close_session(self):
        await self.aiohttp_session.close()


async def fetch_post(session, url, data_json, timeout):
    try:
        await session.post(url, data = data_json, timeout = timeout)
    except asyncio.TimeoutError:
        pass

async def fetch_get(session, url, timeout):
    try:
        await session.get(url, timeout = timeout)
    except asyncio.TimeoutError:
        pass


async def fetch_post_json(session, url, data_json):
    async with session.post(url, data = data_json) as resp:
        return await resp.json(content_type = None) # DISABLING CONTENT TYPE VALIDATION

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
    task = asyncio.create_task(__get_data(session, data, play_url, delay))

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
            pending_tasks.clear()
            return pending.result()
        else:
            return find_last_done(len(pending_tasks)-2, pending_tasks, data)
    else:
        data['action'] = 2
    
    return data


async def __get_data(session, data, url, delay):
    # await asyncio.sleep(delay)

    tmp = get_microsecond()
    # print('Before :' + get_timestamp())
    # r = requests.post(url, data = json.dumps(data))
    # print('After  :' + get_timestamp())

    data = await session.post_expected_json(json.dumps(data), url)


    diff = get_microsecond()-tmp
    if diff < 0:
        diff = 1e6+diff
    session.latency.append(diff/1000)
    

    # data = json.loads(r.text)
    
    # print('Before sleep: ' + get_timestamp())
    
    # print('After sleep: ' + get_timestamp())
    return data

def find_last_done(index, pending_tasks, data):
    if index > -1:
        pending = pending_tasks[index]
        if pending.done():
            for i in range(index+1):
                try:
                    pending_tasks.pop(i)
                except IndexError:
                    data['action'] = 3
                    print('Sa roto la cola')
                    return data
            return pending.result()
        else:
            return find_last_done(index-1, pending_tasks, data)

    else:
        data['action'] = 4
        return data

# def store_in_memory(session, data, timeout):
#     memory_url = session.url + '/memory'
#     try:
#         requests.post(memory_url, data = json.dumps(data), timeout = timeout)
#     except requests.exceptions.Timeout:
#         pass

# def replay_experience(session, timeout):
#     replay_url = session.url + '/replay'
#     try:
#         requests.get(replay_url, timeout = timeout)
#     except requests.exceptions.Timeout:
#         pass


def store_in_memory(session, data, timeout):
    asyncio.run(__store_in_memory(session, data, timeout))

async def __store_in_memory(session, data, timeout):
    memory_url = session.url + '/memory'
    await session.post_json(json.dumps(data), url = memory_url, timeout = timeout)


def replay_experience(session, timeout):
    asyncio.run(__replay_experience(session, timeout))

async def __replay_experience(session, timeout):
    replay_url = session.url + '/replay'
    await session.get_something(url = replay_url, timeout = timeout)

