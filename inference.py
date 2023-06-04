import argparse
import os
import torch
from torch import autocast
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPTokenizer
from utils import get_metrics, process_img, inf_save
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import torchvision
from datetime import datetime
import random
import lpips
import numpy as np

loss_fn = lpips.LPIPS(net='alex')


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
        default="",
        type=str,
        help="Target prompt.",
    )
    parser.add_argument(
        "--gt_image",
        type=str,
        help="chair.png",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--seed",
        default=1011,
        type=int,
        help="Seed",
    )
    parser.add_argument(
        "--image_num",
        type=int,
        default=5,
        help="Seed",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="Seed",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Seed",
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
        args.pretrained_model_name, scheduler=scheduler, torch_dtype=torch.float16, safety_checker=None).to("cuda")
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

    # Concat target and hiper embeddings
    embedding_list=["hotdog_2/hotdog_nhiper30.pt", "hotdog_3/hotdog_nhiper30.pt", "hotdog_4/hotdog_nhiper73.pt", "hotdog_5/hotdog_13_nhiper73.pt"]
    for embedding_ in embedding_list:
        hiper_embeddings = torch.load(f'{embedding_}').to("cuda")
        n_hiper = hiper_embeddings.shape[1]
        inference_embeddings = torch.cat(
            [target_embedding[:, :-n_hiper], hiper_embeddings], 1)

        # Generate target images
        num_samples = 1
        guidance_scale = 7.5
        num_inference_steps = args.num_inference_steps
        height = 512
        width = 512

        with autocast("cuda"), torch.inference_mode():
            model_path = os.path.join(args.output_dir, 'infer')
            os.makedirs(model_path, exist_ok=True)
            for idx, embd in enumerate([inference_embeddings]):
                lpips_list = []
                psnr_list = []
                no_psnr_list = []
                for i in range(args.image_num):
                    seed = random.randrange(1, 10000)
                    set_seed(seed)
                    g_cuda = torch.Generator(device='cuda')
                    g_cuda.manual_seed(seed)

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
                        image1 = images.images[j]
                        image1.save(model_path+f"/infer_img_{i}.png")
                        init_image, input_image = process_img(args.gt_image, 512)
                        input_image = input_image.squeeze()
                        input_image = to_pil_image(input_image)
                        totensor = torchvision.transforms.ToTensor()
                        image = totensor(image1)

                        lpips, psnr, psnr1 = get_metrics(
                            init_image, image, loss_fn)
                        
                        print(f"lpips : {lpips}, PSNR : {psnr1}, No gray PSNR : {psnr}")
                        
                        lpips_list.append(lpips)
                        psnr_list.append(psnr1)
                        no_psnr_list.append(psnr)

                        inf_images = []
                        inf_images.append(input_image)
                        inf_images.append(image1)
                        inf_save(inf_images, [f'GT_Image', f'Const_{round(psnr, 5)}'], model_path+f"/seed_{args.seed}_{i}.png")
                        if i*num_samples+j == args.image_num:
                            break
                with open("./compare.txt", 'a') as file:
                    file.write("\n" + f"image lpips : {np.mean(lpips_list)}, PSNR : {np.mean(psnr_list)}, No gray PSNR : {psnr}")



if __name__ == "__main__":
    main()
