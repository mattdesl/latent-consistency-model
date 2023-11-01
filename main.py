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

        if args.continuous:
            print("Continuous passed, will iterate forever.")

            try:
                while True:
                    for prompt in prompts:
                        print(f"Generating image for prompt: '{prompt}'")
                        output_path = predictor.predict(prompt, args.width, args.height, args.steps, args.seed)
                        print(f"Output image saved to: {output_path}")
            except KeyboardInterrupt:
                print("\nStopped by user.")
        else:
            for prompt in prompts:
                output_path = predictor.predict(prompt, args.width, args.height, args.steps, args.seed)
                print(f"Output image saved to: {output_path}")
    else:
        if not args.prompt:
            print("Please provide a prompt or a prompt file.")
            return

        if args.continuous:
            try:
                while True:
                    output_path = predictor.predict(args.prompt, args.width, args.height, args.steps, args.seed)
                    print(f"Output image saved to: {output_path}")
            except KeyboardInterrupt:
                print("\nStopped by user.")
        elif args.interactive:
            try:
                while True:
                    prompt = input("Enter your image prompt: ")
                    output_path = predictor.predict(prompt, args.width, args.height, args.steps, args.seed)
                    print(f"Output image saved to: {output_path}")
            except KeyboardInterrupt:
                print("\nStopped by user.")
        else:
            predictor.predict(args.prompt, args.width, args.height, args.steps, args.seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images based on text prompts.")
    parser.add_argument("--prompt", type=str, default=None, help="A text prompt for image generation.")
    parser.add_argument("--prompt-file", type=str, default=None, help="A file containing text prompts for image generation, one prompt per line.")
    parser.add_argument("--width", type=int, default=512, help="The width of the generated image.")
    parser.add_argument("--height", type=int, default=512, help="The height of the generated image.")
    parser.add_argument("--steps", type=int, default=8, help="The number of inference steps.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random number generation.")
    parser.add_argument("--continuous", action='store_true', help="Enable continuous generation.")
    parser.add_argument("--interactive", action='store_true', help="Enable interactive mode.")
    return parser.parse_args()

if __name__ == "__main__":
    main()
