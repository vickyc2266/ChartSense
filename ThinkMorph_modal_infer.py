import modal
import torch
from PIL import Image

app = modal.App("thinkmorph-complete")

LOCAL_CODE = "/Users/vc/Desktop/GenAI/ThinkMorph"
LOCAL_WEIGHTS = "/Users/vc/Desktop/GenAI/ThinkMorph/ThinkMorph-7B"

# Modal image
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.05-py3")
    .pip_install(
        "packaging",
        "ninja",
        "wheel",
        "setuptools",
        "numpy<2.0",
        "Pillow",
        "accelerate",
        "opencv-python-headless==4.5.5.64",
        "safetensors",
        "transformers",
        "torchvision",
    )
    .add_local_dir(LOCAL_CODE, "/root/ThinkMorph", copy=True)
    .add_local_dir(LOCAL_WEIGHTS, "/root/ThinkMorph-7B", copy=True)
)

@app.function(image=image, gpu="A10G", timeout=1800)
def run_inference(image_path: str, prompt: str):
    import sys, os
    sys.path.append("/root/ThinkMorph")

    model_path = "/root/ThinkMorph-7B"

    import random
    import numpy as np

    from data.transforms import ImageTransform
    from data.data_utils import pil_img2rgb, add_special_tokens

    from modeling.bagel import (
        BagelConfig, Bagel, Qwen2Config,
        Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
    )
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.bagel.qwen2_navit import NaiveCache
    from modeling.autoencoder import load_ae

    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
    from safetensors.torch import load_file

    llm_config = Qwen2Config.from_json_file(f"{model_path}/llm_config.json")
    vit_config = SiglipVisionConfig.from_json_file(f"{model_path}/vit_config.json")

    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(f"{model_path}/ae.safetensors")

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = infer_auto_device_map(
        model,
        # max_memory={"cuda:0": "30GiB"},
        max_memory={0: "30GiB"},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    for k in [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]:
        # device_map[k] = "cuda:0"
        device_map[k] = 0

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=f"{model_path}/model.safetensors",
        device_map=device_map,
        dtype=torch.bfloat16,
        offload_buffers=True,
        force_hooks=True,
        offload_folder="/tmp/offload",
    )

    from inferencer import InterleaveInferencer
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    inference_hyper = dict(
        max_think_token_n=4096,
        do_sample=True,
        text_temperature=0.3,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
    )

    img = Image.open(image_path)

    outputs = inferencer(
        image=img,
        text=prompt,
        think=True,
        understanding_output=False,
        **inference_hyper,
    )

    final_str = ""
    for item in outputs:
        if isinstance(item, str):
            final_str = item

    return final_str


@app.local_entrypoint()
def main(
    image: str = "/Users/vc/Desktop/GenAI/ThinkMorph/test_images/Visual_Search.jpg",
    prompt: str = "What is the color of the cart?"
):
    out = run_inference.remote(image, prompt)
    print("\n===== THINKMORPH OUTPUT =====")
    print(out)
