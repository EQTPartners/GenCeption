import os
import torch
import argparse
from functools import partial
import logging
from tqdm import tqdm
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from genception.utils import test_sample, encode_image_os, prompt
from genception.file_utils import find_image_files

logging.basicConfig(level=logging.INFO)
torch.backends.cudnn.enabled = False



def get_desc_mPLUG(image, image_processor, lmm_model, tokenizer, prompt, device):
    """
    Given an image, generate a description using the mPLUG model

    Args:
    image: CLIPImageProcessor: The image to describe
    image_processor: callable: The image processor
    lmm_model: The language model
    tokenizer: The tokenizer
    prompt: str: The prompt for the model

    Returns:
    str: The description of the image
    """
    conv = conv_templates["mplug_owl2"].copy()
    max_edge = max(image.size)
    resized_image = image.resize((max_edge, max_edge))
    image_tensor = process_images([resized_image], image_processor)
    if device == "cuda":
        image_tensor = image_tensor.to(lmm_model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(lmm_model.device)
    )
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    temperature = 0.001
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = lmm_model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            attention_mask=attention_mask,
        )

    image_desc = tokenizer.decode(
        output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()
    return image_desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/examples")
    parser.add_argument("--model", type=str, default="MAGAer13/mplug-owl2-llama2-7b")
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.info(args)

    device = args.device
    model_name = get_model_name_from_path(args.model)
    tokenizer, lmm_model, image_processor, _ = load_pretrained_model(
        args.model,
        None,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device=device,
    )
    if device == "cpu":
        lmm_model = lmm_model.float()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    get_desc_function = partial(
        get_desc_mPLUG,
        image_processor=image_processor,
        lmm_model=lmm_model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device
    )
    encode_image_function = encode_image_os

    output_folder = os.path.join(args.dataset, "results_mplug")
    if os.path.exists(args.dataset):
        os.makedirs(output_folder, exist_ok=True)
    else:
        raise FileExistsError(f"{args.dataset} not found!")

    logging.debug("Loaded model. Entered main loop.")
    for img_file in tqdm(find_image_files(args.dataset)):
        try:
            logging.info(img_file)
            test_sample(
                seed_image=img_file,
                n_iteration=args.n_iter,
                output_folder=output_folder,
                get_desc_function=get_desc_function,
                encode_image_function=encode_image_function,
            )
        except Exception as e:
            logging.warning("caught error:")
            logging.warning(e)
            continue


if __name__ == "__main__":
    main()
