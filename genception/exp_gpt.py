import os
import requests
import argparse
from functools import partial
import logging
from tqdm import tqdm
from openai import OpenAI
from genception.utils import test_sample, encode_image_base64, prompt
from genception.file_utils import find_image_files

logging.basicConfig(level=logging.INFO)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
api_key = client.api_key



def get_desc_gpt(image, prompt, model_ver):
    """
    Given an image, generate a description using the gpt-4-vision model

    Args:
    image: Image: The image to describe
    prompt: str: The prompt for the model
    model_ver: the model version

    Returns:
    str: The description of the image
    """
    payload = {
        "model": model_ver,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0,
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    image_desc = response.json()["choices"][0]["message"]["content"]
    return image_desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-4-vision-preview") # use "gpt-4o" to test GPT-4o
    args = parser.parse_args()

    logging.info(args)

    get_desc_function = partial(get_desc_gpt, prompt=prompt, model_ver=args.model)
    encode_image_function = encode_image_base64

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
