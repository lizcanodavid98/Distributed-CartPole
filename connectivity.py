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

    async def close_session(self):
        await self.session.close()

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



async def start_solver(session, observation_space, action_space):
    solver_params = json.dumps({
            'observation-space': observation_space, 
            'action-space': action_space
            })
    
    solver_url = session.url + '/solver'
    await session.post_json(solver_params, url = solver_url)

async def load_model(session, model):
    load_model_url = session.url + '/load'
    await session.post_json(json.dumps({'text':model}), url = load_model_url)

async def get_action(session, data, training = False):
    if training:
        play_url = session.url
    else:
        play_url = session.url + '/play'
    data = await session.post_expected_json(json.dumps(data), url = play_url)
    return data

async def store_in_memory(session, data):
    memory_url = session.url + '/memory'
    await session.post_json(json.dumps(data), url = memory_url)

async def replay_experience(session):
    replay_url = session.url + '/replay'
    await session.get_something(url = replay_url)