import os
import torch
import argparse
from functools import partial
import logging
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from genception.utils import encode_image_os, prompt, test_sample
from genception.file_utils import find_image_files

torch.backends.cudnn.enabled = False


def get_desc_llava(image, lmm_processor, lmm_model, prompt, device):
    """
    Given an image, generate a description using the llava model

    Args:
    image: Image: The image to describe
    lmm_processor: callable: The language model processor
    lmm_model: The language model
    prompt: str: The prompt for the model

    Returns:
    str: The description of the image
    """
    inputs = lmm_processor(text=prompt, images=image, return_tensors="pt").to(device)
    outputs = lmm_model.generate(**inputs, max_new_tokens=512, do_sample=False)
    answer = lmm_processor.batch_decode(outputs, skip_special_tokens=True)[0]
    image_desc = answer.split("ASSISTANT:")[1].strip()
    return image_desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,required=True)
    parser.add_argument("--model", type=str, default="llava7b")
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.info(args)

    device = args.device
    load_in_8bit = True
    if device == "cpu":
        load_in_8bit = False
    lmm_model = LlavaForConditionalGeneration.from_pretrained(
        f"llava-hf/llava-1.5-{args.model[5:]}-hf", load_in_8bit=load_in_8bit
    )
    lmm_processor = AutoProcessor.from_pretrained(
        f"llava-hf/llava-1.5-{args.model[5:]}-hf"
    )
    model_prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
    get_desc_function = partial(
        get_desc_llava,
        lmm_processor=lmm_processor,
        lmm_model=lmm_model,
        prompt=model_prompt,
        device=device
    )
    encode_image_function = encode_image_os

    output_folder = os.path.join(args.dataset, f"results_{args.model}")
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
