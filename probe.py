  
import asyncio
import datetime


async def __test_precission():
    loop = asyncio.get_running_loop()
    end_time = loop.time() + 5.0
    while True:
        print(datetime.datetime.now())
        if (loop.time() + 1.0) >= end_time:
            break
        await asyncio.sleep(0.005)

def test_precission():
    asyncio.run(__test_precission())


def get_timestamp():
    time = datetime.datetime.now()
    return str(time.second) + ':' + str(time.microsecond)

if __name__ == "__main__":
    test_precission()
