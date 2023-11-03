import torch
import numpy as np
import json
import aiohttp
import aiohttp.web
import websockets
import base64
import aiohttp_cors
import os
from predict import Predictor
from PIL import Image
import io
import asyncio
import time
import math

model = None
generator = None
ws = None

async def send_ws_message(ws, message):
    if ws is not None and not ws.closed:
        await ws.send_str(message)
        
def create_image_grid(images, background_color='white'):
    if not images:
        raise ValueError("The images list cannot be empty")

    num_images = len(images)
    # Calculate the number of rows and columns for the grid
    rows = round(num_images ** 0.5)
    cols = math.ceil(num_images / rows)

    # Get dimensions of the first image (assuming all images are the same size)
    img_width, img_height = images[0].size
    
    # Create a new image with specified background color
    grid_img = Image.new('RGB', (img_width * cols, img_height * rows), background_color)
    
    for i, image in enumerate(images):
        col = i % cols
        row = i // cols
        grid_img.paste(image, (img_width * col, img_height * row))

    return grid_img

async def websocket_simple(request):
    global ws
    ws = aiohttp.web.WebSocketResponse()
    print("Connected to ws", ws)
    await ws.prepare(request)
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            print("Received Message", msg)
    return ws

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
              # z = r.latents.clone()
              images = model.latent_to_image(r.latents, fast=True)
              # print("GOT IMAGES", len(images))
              if len(images) > 1:
                 image = create_image_grid(images)
              else:
                 image = images[0]
              # image = images[0]
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
        print("New Generator")
        generator = model.generate(
            prompt_embeds=prompt_embeds,
            width=data['width'],
            height=data['height'],
            steps=data['steps'],
            seed=data['seed'],
            num_images_per_prompt=1,
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

async def generate_handler(request):
    global generator
    # q = request.query
    # width = int(q['width'])
    # height = int(q['height'])
    # seed = int(q.get('seed', int.from_bytes(os.urandom(2), "big")))
    # steps = int(q.get('steps', 4))
    # prompt_embeds = None
    # prompt = None
    # if 'prompt_embeds' in q:
    #     prompt_embeds = q['prompt_embeds']
    # if 'prompt' in q:
    #     prompt = q['prompt']

    data = await request.json()
    seed = int(data.get('seed', int.from_bytes(os.urandom(2), "big")))
    data['seed'] = seed
    if 'prompt_embeds' in data:
        data['prompt_embeds'] = torch.tensor(data['prompt_embeds']).to(model.device)
    generator = model.generate(
        **data,
    )

    return aiohttp.web.json_response({ 'seed': seed })
    # prompt = request.query
    # generator = model.generate(
    #     prompt='flower sunset meadow',
    #     steps=12,
    #     width=256,
    #     height=256,
    #     # intermediate_steps=False
    #     # width=data['width'],
    #     # height=data['height'],
    #     # steps=data['steps'],
    #     # seed=data['seed'],
    #     # num_images_per_prompt=1,
    #     # guidance_scale=data['guidance_scale']
    # )

async def stream_handler(request):
    global generator, ws
    boundary = b'frame'
    response = aiohttp.web.StreamResponse(status=200, reason='OK', headers={'Content-Type': 'multipart/x-mixed-replace; ''boundary=--%s' % boundary,})
    await response.prepare(request)

    async def write_image_chunk (response, data):
        chunk = '--{}\r\n'.format(boundary).encode('utf-8')
        chunk += b'Content-Type: image/jpg\r\n'
        chunk += 'Content-Length: {}\r\n'.format(len(data)).encode('utf-8')
        chunk += b"\r\n"
        chunk += data
        chunk += b"\r\n"
        await response.write(chunk)

    running = True
    while running:
        if generator is not None:
            try:
                r = next(generator)
                fast = r.step != r.steps - 1
                images = model.latent_to_image(r.latents, fast=fast)
                image = images[0]
                bytes = io.BytesIO()
                image.save(bytes, format='JPEG')
                data = bytes.getvalue()
                await write_image_chunk(response, data)
                asyncio.create_task(send_ws_message(ws, "image_chunk"))
                # await response.drain()
                # output_path = model.save_image(image,'00',r.steps,r.step)
                # await response.drain()
                if r.step == r.steps - 1:
                    # await asyncio.sleep(0.1)
                    # print("High Res")
                    # images = model.latent_to_image(r.latents, fast=False)
                    # image = images[0]
                    # bytes = io.BytesIO()
                    # image.save(bytes, format='JPEG')
                    # data = bytes.getvalue()
                    await write_image_chunk(response, data)
                    # print("SAINVG FULL")
                    # cur = torch.nn.functional.interpolate(r.latents, scale_factor=2, mode='nearest')
                    # images = model.latent_to_image(cur, fast=fast)
                    # image = images[0]
                    # model.save_image(image, '00', r.steps, r.step)
                    asyncio.create_task(send_ws_message(ws, "complete"))
                    # output_path = model.save_image(image,'x0',r.steps,r.step)
                    # await response.drain()
            except StopIteration:
                print("Generation complete")
                generator = None
            except ConnectionResetError:
                print("Connection was reset. Stopping image chunk writing.")
                running = False
            except asyncio.CancelledError:
                print("Image chunk writing was cancelled.")
        else:
            pass
        await asyncio.sleep(1.0 / 120.0)
    return response

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

async def image_handler (request):
    q = request.query
    prompt = q.get('prompt', 'space')
    width = int(q.get('width', 256))
    height = int(q.get('height', 256))
    steps = int(q.get('steps', 4))
    seed = int(q.get('seed', int.from_bytes(os.urandom(2), "big")))
    guidance_scale = float(q.get('guidance_scale', 7.5))
    format = (q.get('format', 'png') or 'png').lower()
    contentType = 'image/png' if format == 'png' else 'image/jpeg'
    data = model.run(
        prompt=prompt,
        width=width,
        height=height,
        steps=steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    images = model.latent_to_image(data.latents)
    image = images[0]
    bytes = io.BytesIO()
    image.save(bytes, format='PNG' if format == 'png' else 'JPEG')
    return aiohttp.web.Response(
        body=bytes.getvalue(),
        headers={
            'Content-Type': contentType,
            'x-seed': str(seed),
            'x-steps': str(steps)
        }
    )

async def init_app():
    app = aiohttp.web.Application(client_max_size=10 * 1024 * 1024)
    
    # app.router.add_route('*', '/', root_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/socket', websocket_simple)
    app.router.add_get('/prompt', prompt_handler)
    app.router.add_get('/stream', stream_handler)
    app.router.add_post('/generate', generate_handler)
    app.router.add_get('/image', image_handler)
    app.router.add_static('/', 'static')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
    })
    # Configure CORS on all routes.
    for route in list(app.router.routes()):
        cors.add(route)

    return app

if __name__ == '__main__':
    model = Predictor()
    # generator = model.generate(
    #     prompt='flower sunset meadow',
    #     steps=12,
    #     width=256,
    #     height=256,
    #     # intermediate_steps=False
    #     # width=data['width'],
    #     # height=data['height'],
    #     # steps=data['steps'],
    #     # seed=data['seed'],
    #     # num_images_per_prompt=1,
    #     # guidance_scale=data['guidance_scale']
    # )
    app = aiohttp.web.run_app(init_app(), port=8000)
    print("Server started on http://localhost:8000")