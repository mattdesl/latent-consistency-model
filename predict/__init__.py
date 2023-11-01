import os
import torch
import argparse
import time
import threading
import asyncio
from diffusers import DiffusionPipeline
import signal
# from diffusers.image_processor import VaeImageProcessor
from collections import namedtuple

PredictionResult = namedtuple('PredictionResult', [
    'latents',
    'step',
    'steps',
    'prompt_embeds'
])

class Predictor:
    def __init__(self):
        self.pipe = self._load_model()
        self.device = self.pipe._execution_device

    def _load_model(self):
        model = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main",
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )
        model.to(torch_device="cpu", torch_dtype=torch.float32).to('mps:0')
        return model
    
    def run_generate (self, prompt, seed=None, **kwargs):
        seed = seed or int.from_bytes(os.urandom(2), "big")
        for data in self.generate(prompt=prompt, seed=seed, **kwargs):
            images = self.latent_to_image(data[0])
            image = images[0]
            output_path = self.save_result(image,prompt,seed,data[1])
            print(f"{data[1]+1} of {data[2]} image saved to: {output_path}")

    def run (self, prompt, seed=None, **kwargs):
        seed = seed or int.from_bytes(os.urandom(2), "big")
        for data in self.generate(prompt=prompt, seed=seed, **kwargs):
            if data[1] == data[2] - 1:
                return data

    def generate (self, 
                prompt = None,
                seed = None,
                steps = 4,
                prompt_embeds = None,
                **kwargs):
        seed = seed or int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)
        device = self.device
        if prompt_embeds is None:
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device, 1, prompt_embeds=None
            )
        yield from self.do_predict(
            prompt_embeds=prompt_embeds,
            lcm_origin_steps=50,
            **kwargs,
            num_inference_steps=steps
        )

    def encode_prompt (self, prompt):
        device = self.device
        return self.pipe._encode_prompt(
            prompt, device, 1, prompt_embeds=None
        )

    def do_predict (self,
        prompt = None,
        height = 768,
        width = 768,
        guidance_scale = 7.5,
        num_images_per_prompt = 1,
        latents = None,
        num_inference_steps = 4, 
        lcm_origin_steps = 50,
        prompt_embeds = None,
        cross_attention_kwargs = None,
        intermediate_steps = True
        ):
        pipe = self.pipe
        
        # 0. Default height and width to unet
        height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self.device
        
        # 3. Encode input prompt
        prompt_embeds = pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            prompt_embeds=prompt_embeds,
        )
        
        # 4. Prepare timesteps
        pipe.scheduler.set_timesteps(num_inference_steps, lcm_origin_steps)
        timesteps = pipe.scheduler.timesteps
        
        # 5. Prepare latent variable
        num_channels_latents = pipe.unet.config.in_channels
        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            latents,
        )
        
        bs = batch_size * num_images_per_prompt
        
        # 6. Get Guidance Scale Embedding
        w = torch.tensor(guidance_scale).repeat(bs)
        w_embedding = pipe.get_w_embedding(w, embedding_dim=256).to(device)
        
        # 7. LCM MultiStep Sampling Loop:
        for i, t in enumerate(timesteps):
            print(i, t)
            ts = torch.full((bs,), t, device=device, dtype=torch.long)
            
            # model prediction (v-prediction, eps, x)
            model_pred = pipe.unet(
                latents,
                ts,
                timestep_cond=w_embedding,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs, 
                return_dict=False)[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents, denoised = pipe.scheduler.step(model_pred, i, t, latents, return_dict=False)
            
            should_ret = intermediate_steps or i == num_inference_steps - 1
            if should_ret:
                yield PredictionResult(denoised, i, num_inference_steps, prompt_embeds)
    
    def latent_to_image (self, latent):
        pipe = self.pipe
        image = pipe.vae.decode(latent / pipe.vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = torch.stack(
            [pipe.image_processor.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])]
        )
        image = pipe.image_processor.pt_to_numpy(image.detach())
        image = pipe.image_processor.numpy_to_pil(image)
        return image

    def save_result(self, result, prompt: str, seed: int, steps: int):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Make prompt filesystem safe
        prompt = prompt.replace("\n", "__")
        prompt = prompt.replace(" ", "_")
        prompt = prompt.replace("/", "_")
        prompt = prompt.replace("\\", "_")
        prompt = prompt.replace(":", "_")
        prompt = prompt.replace("*", "_")
        prompt = prompt.replace("?", "_")
        prompt = prompt.replace("\"", "_")
        prompt = prompt.replace("<", "_")
        prompt = prompt.replace(">", "_")
        prompt = prompt.replace("|", "_")

        output_path = os.path.join(output_dir, f"{prompt}-time-{timestamp}-seed-{seed}-steps-{steps}.png")
        result.save(output_path)
        # print(f"Output image saved to: {output_path}")
        return output_path
