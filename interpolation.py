import argparse
import os
import torch
from torch import autocast
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from slerp import slerp

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The directory where the target embeddings are saved.",
    )
    parser.add_argument(
        "--target_txt",
        type=str,
        help="Target prompt.",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed",
        default=1000
    )
    parser.add_argument(
        "--image_num",
        type=int,
        help="Seed",
        default=1
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        g_cuda = torch.Generator(device='cuda')
        g_cuda.manual_seed(args.seed)

    # Load pretrained models
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name, subfolder="tokenizer", use_auth_token=True)
    CLIP_text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name, subfolder="text_encoder", use_auth_token=True)

    # Encode the target text.
    text_ids_tgt = tokenizer(args.target_txt, padding="max_length", truncation=True,
                             max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    CLIP_text_encoder.to('cuda', dtype=torch.float32)
    with torch.inference_mode():
        target_embedding = CLIP_text_encoder(
            text_ids_tgt.to('cuda'))[0].to('cuda')
    del CLIP_text_encoder

    # hiper_embeddings = torch.load(f'mic1.pt').to("cuda")
    hiper_embeddings = torch.load(f'mic_fixed103.pt').to("cuda")
    hiper_embeddings2 = torch.load(f'fixed_100_to_112.pt').to("cuda")

    divided_num = 30
    for intp in range(1, divided_num):
        # Concat target and hiper embeddings
        print(intp)
        n_hiper = hiper_embeddings.shape[1]
        inter_hiper_embeddings = (
            hiper_embeddings*intp + hiper_embeddings2*(divided_num-intp))/divided_num

        # inter_hiper_embeddings = slerp(hiper_embeddings, hiper_embeddings2, intp/divided_num)
        # print("ㅅ수ㅔ입", hiper_embeddings.shape, hiper_embeddings[0].shape)
        # for i in range(hiper_embeddings.shape[1]):
        # print(hiper_embeddings[0][i].shape)
        # hiper_embeddings[0][i] = (hiper_embeddings[0][i] + hiper_embeddings2[0][i])/2
        inter_hiper_embeddings.to("cuda")
        print("shape - is this well interpolated? ", inter_hiper_embeddings.shape)

        inference_embeddings = torch.cat(
            [target_embedding[:, :-n_hiper], inter_hiper_embeddings], 1)

        # Generate target images
        num_samples = 5
        guidance_scale = 7.5
        num_inference_steps = 50
        height = 512
        width = 512

        with autocast("cuda"), torch.inference_mode():
            model_path = os.path.join(args.output_dir, 'inferences')
            os.makedirs(model_path, exist_ok=True)
            for idx, embd in enumerate([inference_embeddings]):
                # for i in range(args.image_num//num_samples+1):
                for i in range(args.image_num):
                    images = pipe(
                        prompt_embeds=embd,
                        height=height,
                        width=width,
                        num_images_per_prompt=num_samples,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=g_cuda
                    )
                    for j in range(len(images)):
                        if images.nsfw_content_detected[j]:
                            continue
                        image = images.images[j]
                        image.save(model_path+f'/{intp}_{args.target_txt}_{i}.png')
                        break
                        if i*num_samples+j == args.image_num:
                            break


if __name__ == "__main__":
    main()
