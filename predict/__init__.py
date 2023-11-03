import os
import torch
import time
from diffusers import DiffusionPipeline, AutoencoderTiny
# import signal
# import compel
from collections import namedtuple

PredictionResult = namedtuple('PredictionResult', [
    'latents',
    'step',
    'steps',
    'prompt_embeds'
])

class Predictor:
    def __init__(self, with_fast = True):
        self.pipe = self._load_model()
        self.device = self.pipe._execution_device
        
        self.pipe.set_progress_bar_config(disable=True)
        # self.compel_proc = Compel(
        #     tokenizer=self.pipe.tokenizer,
        #     text_encoder=self.pipe.text_encoder,
        #     truncate_long_prompts=False,
        # )
        if with_fast:
            self.pipe.fast_vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd", torch_dtype=torch.float32, use_safetensors=True
            ).to(self.device)
        self.do_predict = torch.compile(self._do_predict)


    def _load_model(self):
        model = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline="latent_consistency_txt2img",
            # custom_pipeline="latent_consistency_img2img",
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

    def run (self, seed=None, **kwargs):
        seed = seed or int.from_bytes(os.urandom(2), "big")
        for data in self.generate(seed=seed, **kwargs, intermediate_steps=False):
            if data.step == data.steps - 1:
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
        # device = self.device
        if prompt_embeds is None:
            prompt_embeds = self.encode_prompt(prompt)
        yield from self._do_predict(
            lcm_origin_steps=50,
            **kwargs,
            prompt_embeds=prompt_embeds,
            num_inference_steps=steps
        )

    @torch.no_grad()
    def encode_prompt (self, prompt):
        # return self.compel_proc(prompt)
        device = self.device
        return self.pipe._encode_prompt(
            prompt, device, 1, prompt_embeds=None
        )
    
    @torch.no_grad()
    def _do_predict (self,
        prompt = None,
        height = 512,
        width = 512,
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
        # prompt_embeds = torchvision.transforms.functional.gaussian_blur(prompt_embeds, kernel_size=3, sigma=0.4)
        # prompt_embeds = torch.nn.functional.interpolate(prompt_embeds, scale_factor=0.5, mode='nearest')
        # prompt_embeds = torch.nn.functional.interpolate(prompt_embeds, scale_factor=2, mode='nearest')

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

        # print(latents.size())
        # latents = torch.nn.functional.interpolate(latents, scale_factor=0.5, mode='bilinear')
        # latents = torch.nn.functional.interpolate(latents, scale_factor=0, mode='bilinear')
        
        bs = batch_size * num_images_per_prompt
        
        # 6. Get Guidance Scale Embedding
        w = torch.tensor(guidance_scale).repeat(bs)
        w_embedding = pipe.get_w_embedding(w, embedding_dim=256).to(device)
        
        # 7. LCM MultiStep Sampling Loop:
        for i, t in enumerate(timesteps):
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
            # latents = torchvision.transforms.functional.gaussian_blur(latents, kernel_size=5, sigma=0.2)

            should_ret = intermediate_steps or i == num_inference_steps - 1
            if should_ret:
                yield PredictionResult(denoised, i, num_inference_steps, prompt_embeds)
    
    
    # def pixelate_image(image_array, pixelation_level):
    #     # Convert the image array to a PIL Image
    #     image = Image.fromarray(image_array)

    #     # Calculate the new dimensions
    #     width, height = image.size
    #     small_width = width // pixelation_level
    #     small_height = height // pixelation_level

    #     # Resize down using NEAREST to achieve pixelation
    #     small_image = image.resize((small_width, small_height), resample=Image.NEAREST)

    #     # Resize back to original size
    #     pixelated_image = small_image.resize((width, height), resample=Image.NEAREST)

    #     # Convert back to array if needed
    #     pixelated_array = np.array(pixelated_image)

    #     return pixelated_array

    @torch.no_grad()
    def latent_to_image (self, latent, fast=False):
        pipe = self.pipe
        # latent += latent
        
        # print(latent[0].size())
        # latent[0]
        # size = latent.size() / 2
        # scaler = 8
        # sf = 1/(2)
        # latent = torch.nn.functional.interpolate(latent, scale_factor=sf, mode='nearest')
        # latent = torch.nn.functional.interpolate(latent, scale_factor=1.0/sf, mode='nearest')
        # latent = torchvision.transforms.functional.gaussian_blur(latent, kernel_size=5, sigma=0.5)
        # latent = torchvision.transforms.functional.adjust_sharpness(latent, 0.5)

        vae = self.pipe.fast_vae if fast else self.pipe.vae
        image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
        do_denormalize = [True] * image.shape[0]
        image = torch.stack(
            [pipe.image_processor.denormalize(image[i]) if do_denormalize[i] else image[i] for i in range(image.shape[0])]
        )
        image = pipe.image_processor.pt_to_numpy(image.detach())
        image = pipe.image_processor.numpy_to_pil(image)
        return image

    def save_image(self, result, seed, steps, i=0):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{timestamp}-seed-{seed}-steps-{steps}-i-{i}.png")
        result.save(output_path)
        print(f"Output image saved to: {output_path}")
        return output_path

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
