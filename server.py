import torch
import numpy as np
import json
import aiohttp
import aiohttp.web
import websockets
import base64
import os
from predict import Predictor
from PIL import Image
import io
import asyncio
import time
model = None
generator = None

async def websocket_handler(request):
    # max_size=1024 * 1024 * 1024
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)

    async def ping (ws):
      global generator
      try:
        if generator is not None:
          try:
              r = next(generator)
              z = r.latents.clone()
              images = model.latent_to_image(z)
              image = images[0]
              bytes = io.BytesIO()
              image.save(bytes, format='PNG')

              # start = time.perf_counter()
              # latents_list = r.latents.tolist()
              # prompt_embeds_list = r.prompt_embeds.tolist()
              # data = {
              #     'image': 'data:image/png;base64,' + base64.b64encode(bytes.getvalue()).decode('utf-8'),
              #     'latents': latents_list,
              #     'prompt_embeds': prompt_embeds_list
              # }
              # # print('sending str')
              # await ws.send_str(json.dumps(data))
              # end = time.perf_counter()
              # print(f"Elapsed Time: {end-start} seconds")
              # print('finished')
              await ws.send_bytes(bytes.getvalue())
          except StopIteration:
              print("Generation completed")
              generator = None
        else:
            pass
      except asyncio.CancelledError:
         print("Task cancelled")
  
    def run (data):
        global generator
        if generator is not None:
            generator.close()
        if 'prompt_embeds' in data:
            prompt_embeds = torch.tensor(data['prompt_embeds']).to(model.device)
            # prompt_embeds = torch.as_tensor(data['prompt_embeds']).to(model.device)
        else:
            prompt_embeds = model.encode_prompt(data['prompt'])
        generator = model.generate(
            prompt_embeds=prompt_embeds,
            width=data['width'],
            height=data['height'],
            steps=data['steps'],
            seed=data['seed'],
            guidance_scale=data['guidance_scale']
        )
        return prompt_embeds

    async def send_prompt_embeds (prompt_embeds):
       await ws.send_json({
            'prompt_embeds': prompt_embeds.tolist()
        })
       
    loop = asyncio.get_event_loop()
    tasks = set()

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == 'ping':
              task = loop.create_task(ping(ws))
              tasks.add(task)
              task.add_done_callback(tasks.discard)
            else:
              data = msg.json()
              if 'prompt' in data or 'prompt_embeds' in data:
                print('New Prompt')
                for task in tasks:
                   task.cancel()
                tasks.clear()
                run(data)
                # task = loop.create_task(send_prompt_embeds(prompt_embeds))
                # tasks.add(task)
                # task.add_done_callback(tasks.discard)
                

    return ws

async def root_handler(request):
    return aiohttp.web.HTTPFound('/index.html')

# async def generate_handler(request):
#     prompt = request.query.get('prompt')
#     prompt_embeds = model.encode_prompt(prompt)
#     prompt_embeds_list = prompt_embeds.tolist()
#     return aiohttp.web.json_response(prompt_embeds_list)

async def prompt_handler(request):
    prompt = request.query.get('prompt')
    prompt_embeds = model.encode_prompt(prompt)
    prompt_embeds_list = prompt_embeds.tolist()
    return aiohttp.web.json_response(prompt_embeds_list)

async def init_app():
    app = aiohttp.web.Application()
    app.router.add_route('*', '/', root_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/prompt', prompt_handler)
    # app.router.add_get('/generate', generate_handler)
    app.router.add_static('/', 'static')
    return app

if __name__ == '__main__':
    model = Predictor()
    app = aiohttp.web.run_app(init_app(), port=8000)
    print("Server started on http://localhost:8000")