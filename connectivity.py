import aiohttp, asyncio, nest_asyncio, async_timeout
import json

class mySession:
    def __init__(self, url):
        self.url = url
        self.session = aiohttp.ClientSession()
        nest_asyncio.apply()

    async def post_json(self, data_json, url):
        await fetch_post(self.session, url, data_json)

    async def post_expected_json(self, data_json, url):
        data = await fetch_post_json(self.session, url, data_json)
        return data

    async def get_something(self, url):
        await fetch_get(self.session, url)

    def close_session(self):
        asyncio.run(self.__close_session())

    async def __close_session(self):
        asyncio.run(self.session.close())

def new_session(url):
    s = mySession(url)
    return s

async def fetch_post(session, url, data_json):
    async with session.post(url, data = data_json) as resp:
        pass
            
async def fetch_post_json(session, url, data_json):
    async with session.post(url, data = data_json) as resp:
        return await resp.json()
        
async def fetch_get(session, url):
    async with session.get(url) as resp:
        pass

def start_solver(session, observation_space, action_space):
    asyncio.run(__start_solver(session, observation_space, action_space))

async def __start_solver(session, observation_space, action_space):
    solver_params = json.dumps({
            'observation-space': observation_space, 
            'action-space': action_space
            })
    
    solver_url = session.url + '/solver'
    await session.post_json(solver_params, url = solver_url)


def load_model(session, model):
    asyncio.run(__load_model(session, model))

async def __load_model(session, model):
    load_model_url = session.url + '/load'
    await session.post_json(json.dumps({'text':model}), url = load_model_url)


def get_action(session, data, training = False):
    return asyncio.run(__get_action(session, data, training))

async def __get_action(session, data, training):
    if training:
        play_url = session.url
    else:
        play_url = session.url + '/play'
    data = await session.post_expected_json(json.dumps(data), url = play_url)
    return data


def store_in_memory(session, data):
    asyncio.run(__store_in_memory(session, data))

async def __store_in_memory(session, data):
    memory_url = session.url + '/memory'
    await session.post_json(json.dumps(data), url = memory_url)


def replay_experience(session):
    asyncio.run(__replay_experience(session))

async def __replay_experience(session):
    replay_url = session.url + '/replay'
    await session.get_something(url = replay_url)