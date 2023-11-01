import torch
# import os
# import torch
import argparse
# import time
# from diffusers import DiffusionPipeline
# from diffusers.image_processor import VaeImageProcessor
from predict import Predictor

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
            prompt_a = "wilflower meadow"
            prompt_b = "blood orange sky"
            data_a = predictor.run(
                prompt=prompt_a,
                width=args.width,
                height=args.height,
                steps=args.steps,
                seed=args.seed,
                guidance_scale=7.5,
            )

            # data_b = predictor.run(
            #     prompt=prompt_b,
            #     width=args.width,
            #     height=args.height,
            #     steps=args.steps,
            #     seed=args.seed,
            #     guidance_scale=7.5,
            # )
            
            # xr = data_a.shape[1]
            # print(data_a[0].size())
            # new_latent = torch.lerp(data_a[0], data_b[0], 0.25)
            # new_latent = torch.lerp(data_a[0], data_b[0], 0.25)
            images = predictor.latent_to_image(data_a[0])
            output_path = predictor.save_result(images[0],prompt,args.seed,args.steps)
            print(f"Output image saved to: {output_path}")
    else:
        raise "Needs prompt file"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images based on text prompts.")
    parser.add_argument("--prompt-file", type=str, default=None, help="A file containing text prompts for image generation, one prompt per line.")
    parser.add_argument("--width", type=int, default=256, help="The width of the generated image.")
    parser.add_argument("--height", type=int, default=256, help="The height of the generated image.")
    parser.add_argument("--steps", type=int, default=5, help="The number of inference steps.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation.")
    return parser.parse_args()

if __name__ == "__main__":
    main()
