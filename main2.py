import torch
# import os
import time
# import torch
import argparse
import math
# import time
# from diffusers import DiffusionPipeline
# from diffusers.image_processor import VaeImageProcessor
from predict import Predictor

# def add_echo(audio_tensor, sample_rate, delay_sec, decay):
#     delay_samples = int(sample_rate * delay_sec)
#     echo = torch.cat((torch.zeros(delay_samples), audio_tensor * decay))
#     return audio_tensor + echo[:audio_tensor.size(0)]

def add_echo_to_embedding(embedding_tensor, delay_samples, decay):
    """
    Apply an echo effect to a 1D tensor.

    Parameters:
    embedding_tensor (torch.Tensor): The 1D tensor to apply the echo to.
    delay_samples (int): The delay of the echo in number of samples (elements in the tensor).
    decay (float): The decay factor of the echo. Should be between 0 and 1.

    Returns:
    torch.Tensor: The tensor with the echo applied.
    """
    if delay_samples < 1:
        return embedding_tensor
    
    echo = torch.cat((torch.zeros(delay_samples), embedding_tensor * decay))
    return embedding_tensor + echo[:embedding_tensor.size(0)]

def reflect_across_vector(embedding, vector):
    # Normalize the vector to have unit length
    vector = vector / torch.norm(vector)
    
    # Project the embedding onto the vector
    projection = torch.dot(embedding, vector) * vector
    
    # Reflect the embedding across the vector
    reflection = 2 * projection - embedding
    
    return reflection

def main():
    args = parse_args()
    predictor = Predictor()

    # If prompt file is provided, read prompts from file
    if args.prompt_file:
        prompts = []
        with open(args.prompt_file, "r") as f:
            prompt_lines = f.readlines()
            prompts = [prompt.strip() for prompt in prompt_lines]
            prompts = [prompt.replace(' / ', '\n') for prompt in prompts]

        for prompt in prompts:
            data = predictor.run(
                prompt=prompt,
                width=args.width,
                height=args.height,
                steps=args.steps,
                seed=args.seed,
                guidance_scale=7.5,
            )
            images = predictor.latent_to_image(data[0])
            output_path = predictor.save_result(images[0],prompt,args.seed,args.steps)
            print(f"Output image saved to: {output_path}")
    else:
        # prompt_embeds0 = predictor.encode_prompt('photo of a cute poodle dog, 8k hd, bokeh')
        # embeds = predictor.encode_prompt('painting of a landscape')
        # prompt_embeds1 = predictor.encode_prompt('neon sunset cyberpunk')

        start = time.perf_counter()
        # embeds = predictor.encode_prompt('moon landing photo')
        # embeds = predictor.encode_prompt('neon sunset cyberpunk')
        generator = predictor.generate(
            # prompt_embeds=embeds,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
        )
        for i, data in enumerate(generator):
            images = predictor.latent_to_image(data.latents, fast=False)
            end = time.perf_counter()
            print(f"Time taken: {end-start} seconds")
            # output_path = predictor.save_image(images[0],args.seed,args.steps, i)

            # cur = torch.nn.functional.interpolate(data.latents, scale_factor=1, mode='nearest-exact')
            images = predictor.latent_to_image(cur, fast=False)
            image = images[0]
            predictor.save_image(image, args.seed, data.steps, data.step)
        
        # latents = data.latents
        # count = latents.size()[1]
        # for k in range(count):
        #     sf = 1 / (1+k)
        #     # cur = torch.nn.functional.interpolate(latents, scale_factor=sf, mode='nearest')
        #     # cur = torch.nn.functional.interpolate(cur, scale_factor=1/sf, mode='nearest')
        #     # cur = torch.flip(latents,dims=(1,))
        #     cur = torch.clone(latents)
        #     # # cur = latents[0][k]
        #     # off = 2 * torch.rand(1).to(predictor.device) - 1
        #     # for i in range(count):
        #     #     cur[0][i] += off

        #     z = cur[0]
        #     dims = z.size()

        #     # Define the start indices for each dimension
        #     start_channel, start_row, start_col = 0, 10, 10

        #     # Define the size of the region you want to extract
        #     region_size = (4, 8, 8)

        #     # Extract the region
        #     # cur[0] = cur[0][start_channel:start_channel + region_size[0],
        #     #                 start_row:start_row + region_size[1],
        #     #                 start_col:start_col + region_size[2]]
        #     # cur = torch.tile(cur, (0,1))
        #     # rect_x0 = 32
        #     # rect_y0 = 0
        #     # rect_x1 = 64
        #     # rect_y1 = 32
        #     # for c in range(dims[0]):
        #     #     for y in range(dims[1]):
        #     #         for x in range(dims[0]):
        #     #             if x >= rect_x0 and y >= rect_y0 and x < rect_x1 and y < rect_y1:
        #     #                 z[c][y][x] *= 0
                        

        #     images = predictor.latent_to_image(cur)
        #     output_path = predictor.save_image(images[0],args.seed,args.steps,k)
        

        # count = 5
        # for i in range(count):
        #     t = i / (count - 1) if count > 1 else 0
        #     embeds = torch.clone(prompt_embeds0)
        #     # mirror_vector = torch.randn(embeds.shape)  # Random vector for demonstration
        #     # for k in range(math.floor(embeds.size()[1] * t)):
        #     #     # embeds[0][k] = torch.randn(embeds[0][k].size()[0])
        #     #     embeds[0][k] = prompt_embeds1[0][k]
        #     # embeds = torch.flip(embeds, [1,1])

        #     # for k in range(embeds.size()[1]):
        #     #     embeds[0][k] = embeds[0][k]
            
        #     # embeds = embeds.to(predictor.device)
            
        #     # start_row = 5
        #     # end_row = 10
        #     # torch.manual_seed(args.seed)
        #     # rand = torch.randn(embeds.size()).to(predictor.device)
        #     # embeds = torch.nn.functional.normalize(embeds)
        #     # for row in range(start_row, end_row):
        #         # embeds[0][row] = torch.lerp(embeds[0][row], rand[0][row], t)
        #         # embeds[0][row] = torch.lerp(prompt_embeds0[0][row], prompt_embeds1[0][row], t)
        #     embeds = torch.lerp(prompt_embeds0, prompt_embeds1, t)
        #     data = predictor.run(
        #         prompt_embeds=embeds,
        #         width=args.width,
        #         height=args.height,
        #         steps=args.steps,
        #         seed=args.seed,
        #     )
        #     fast = i == 0
        #     images = predictor.latent_to_image(data[0], fast)
        #     output_path = predictor.save_image(images[0],args.seed,args.steps, i)
        
        # raise "Needs prompt file"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images based on text prompts.")
    parser.add_argument("--prompt-file", type=str, default=None, help="A file containing text prompts for image generation, one prompt per line.")
    parser.add_argument("--prompt", type=str, default=None, help="A prompt.")
    parser.add_argument("--width", type=int, default=256, help="The width of the generated image.")
    parser.add_argument("--height", type=int, default=256, help="The height of the generated image.")
    parser.add_argument("--steps", type=int, default=5, help="The number of inference steps.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation.")
    return parser.parse_args()

if __name__ == "__main__":
    main()
