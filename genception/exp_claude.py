import os
import argparse
from functools import partial
import logging
from tqdm import tqdm
import anthropic
from genception.utils import test_sample, encode_image_base64, prompt
from genception.file_utils import find_image_files

logging.basicConfig(level=logging.INFO)

# defaults to os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic()
api_key = client.api_key


def get_desc_claude(image, prompt, model_spec):
    """
    Given an image, generate a description using the Claude model

    Args:
    image: Image: The image to describe
    prompt: str: The prompt for the model

    Returns:
    str: The description of the image
    """
    message = client.messages.create(
        model=model_spec,
        max_tokens=512,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    image_desc = message.content[0].text

    return image_desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="claude-3-opus-20240229")
    parser.add_argument("--n_iter", type=int, default=3)
    args = parser.parse_args()

    logging.info(args)

    get_desc_function = partial(get_desc_claude, prompt=prompt, model_spec=args.model)
    encode_image_function = encode_image_base64

    output_folder = os.path.join(args.dataset, "results_claude")
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
