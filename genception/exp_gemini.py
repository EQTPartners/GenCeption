import os
import argparse
from functools import partial
import logging
from tqdm import tqdm
from genception.utils import test_sample, prompt
from genception.file_utils import find_image_files

# Install with pip3 install --upgrade google-cloud-aiplatform
# Also need to authorize: gcloud auth application-default login
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image
PROJECT_ID = "PROJECT_ID" # Replace with your Project ID
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

logging.basicConfig(level=logging.INFO)


def get_desc_claude(image, prompt, model_spec):
    """
    Given an image, generate a description using the Claude model

    Args:
    image: Image: The image to describe
    prompt: str: The prompt for the model

    Returns:
    str: The description of the image
    """
    generative_multimodal_model = GenerativeModel(
        model_spec, 
        generation_config={"temperature":.0}
    )
    response = generative_multimodal_model.generate_content(
        [prompt, image]
    )

    image_desc = response.candidates[0].content.parts[0].text

    return image_desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemini-1.5-pro-preview-0409")
    parser.add_argument("--n_iter", type=int, default=3)
    args = parser.parse_args()

    logging.info(args)

    get_desc_function = partial(get_desc_claude, prompt=prompt, model_spec=args.model)
    encode_image_function = Image.load_from_file

    output_folder = os.path.join(args.dataset, "results_gemini")
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
                sleep_between_iterations=10,
            )
        except Exception as e:
            logging.warning("caught error:")
            logging.warning(e)
            continue


if __name__ == "__main__":
    main()
