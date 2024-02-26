import os
import torch
import base64
import pickle
import requests
import argparse
import nltk
from nltk.tokenize import word_tokenize
from functools import partial
from transformers import ViTImageProcessor, ViTModel
from transformers import AutoProcessor, LlavaForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import logging
from tqdm import tqdm
from openai import OpenAI
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from genception.utils import find_files

logging.basicConfig(level=logging.INFO)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
api_key = client.api_key
nltk.download("punkt")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.enabled = False

# VIT model
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")


def image_embedding(image_file: str) -> list[float]:
    """
    Generates an image embedding using a vit model

    Args:
    image_file: str: The path to the image file

    Returns:
    list[float]: The image embedding
    """
    image = Image.open(image_file).convert("RGB")
    inputs = vit_processor(images=image, return_tensors="pt")
    outputs = vit_model(**inputs)
    return outputs.last_hidden_state.tolist()[0][0]


def save_image_from_url(url: str, filename: str):
    """
    Save an image from a given URL to a file

    Args:
    url: str: The URL of the image
    filename: str: The name of the file to save the image to
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
    else:
        logging.warning(
            f"Failed to download image. Status code: {response.status_code}"
        )


def find_image_files(folder_path: str) -> list[str]:
    image_extensions = {".jpg", ".png"}
    return find_files(folder_path, image_extensions)


def count_words(text):
    words = word_tokenize(text)
    return len(words)


def encode_image_os(image_path: str):
    image = Image.open(image_path).convert("RGB")
    return image


def encode_image_gpt4v(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_xt(
    image_desc: str, output_folder: str, i: int, file_name: str, file_extension: str
) -> str:
    """
    Generate an image based on a description using dall-e and save it to a file

    Args:
    image_desc: str: The description of the image
    output_folder: str: The path to the folder to save the image to
    i: int: The iteration number
    file_name: str: The name of the file
    file_extension: str: The extension of the file

    Returns:
    str: The path to the saved image file
    """
    response = client.images.generate(
        model="dall-e-3",
        prompt="Generate an image that fully and precisely reflects this description: {}".format(
            image_desc
        ),
        size="1024x1024",
        quality="standard",
        n=1,
    )
    new_image_filename = os.path.join(
        output_folder, f"{file_name}_{i}.{file_extension}"
    )
    save_image_from_url(response.data[0].url, new_image_filename)
    return new_image_filename


def get_desc_mPLUG(image, image_processor, lmm_model, tokenizer, prompt):
    """
    Given an image, generate a description using the mPLUG model

    Args:
    image: Image: The image to describe
    image_processor: callable: The image processor
    lmm_model: The language model
    tokenizer: The tokenizer
    prompt: str: The prompt for the model

    Returns:
    str: The description of the image
    """
    conv = conv_templates["mplug_owl2"].copy()
    max_edge = max(image.size)
    image = image.resize((max_edge, max_edge))
    image_tensor = process_images([image], image_processor)
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


def get_desc_llava(image, lmm_processor, lmm_model, prompt):
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


def get_desc_gpt4v(image, prompt):
    """
    Given an image, generate a description using the gpt-4-vision model

    Args:
    image: Image: The image to describe
    prompt: str: The prompt for the model

    Returns:
    str: The description of the image
    """
    payload = {
        "model": "gpt-4-vision-preview",
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


def test_sample(
    seed_image: str,
    n_iteration: int,
    output_folder: str,
    get_desc_function: callable,
    encode_image_function: callable,
):
    """
    Iteratively generates T (n_iterations) descriptions and images based on the seed image

    Args:
    seed_image: str: The path to the seed image
    n_iteration: int: The number of iterations to perform
    output_folder: str: The path to the folder to save the results
    get_desc_function: callable: The function to generate the description
    encode_image_function: callable: The function to encode the image
    """
    list_of_desc = []
    list_of_image = []
    list_of_image_embedding = [image_embedding(seed_image)]
    list_of_cos_sim = [1.0]

    current_image_path = seed_image
    current_image_name = os.path.basename(current_image_path)
    file_name, file_extension = current_image_name.split(".")
    logging.debug(f"Image: {current_image_path}")
    pkl_file = os.path.join(output_folder, f"{file_name}_result.pkl")
    if os.path.exists(pkl_file):
        logging.info("Results already exist, skipping")
        return None

    for i in range(n_iteration):
        # Encode the current image and get the description
        image = encode_image_function(current_image_path)
        image_desc = get_desc_function(image)
        list_of_desc.append(image_desc)
        logging.debug(image_desc)

        # generate X^t, append image and embedding
        new_image_filename = generate_xt(
            image_desc, output_folder, i, file_name, file_extension
        )
        list_of_image.append(new_image_filename)
        list_of_image_embedding.append(image_embedding(new_image_filename))

        # Calculate Cosine Sim to original image
        similarity = cosine_similarity(
            [list_of_image_embedding[0]], [list_of_image_embedding[-1]]
        )[0][0]
        list_of_cos_sim.append(similarity)
        logging.info(f"({count_words(image_desc)}, {round(similarity,2)})")

        # Save checkpoint to avoid losing results
        data_to_save = {
            "descriptions": list_of_desc,
            "images": list_of_image,
            "image_embeddings": list_of_image_embedding,
            "cosine_similarities": list_of_cos_sim,
        }
        with open(pkl_file, "wb") as file:
            pickle.dump(data_to_save, file)

        # Update current_image_path for the next iteration
        current_image_path = new_image_filename

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mme_data/color")
    parser.add_argument("--model", type=str, default="llava7b")
    parser.add_argument("--n_iter", type=int, default=5)
    args = parser.parse_args()

    logging.info(args)

    prompt = "Please write a clear, precise, detailed, and concise description of all elements in the image. Focus on accurately depicting various aspects, including but not limited to the colors, shapes, positions, styles, texts and the relationships between different objects and subjects in the image. Your description should be thorough enough to guide a professional in recreating this image solely based on your textual representation. Remember, only include descriptive texts that directly pertain to the contents of the image. You must complete the description using less than 500 words."

    if "llava" in args.model:
        lmm_model = LlavaForConditionalGeneration.from_pretrained(
            f"llava-hf/llava-1.5-{args.model[5:]}-hf", load_in_8bit=True
        )
        lmm_processor = AutoProcessor.from_pretrained(
            f"llava-hf/llava-1.5-{args.model[5:]}-hf"
        )
        prompt = f"<image>\nUSER: {prompt}\nASSISTANT:"
        get_desc_function = partial(get_desc_llava, lmm_processor, lmm_model, prompt)
        encode_image_function = encode_image_os
    elif args.model == "mPLUG":
        model_path = "MAGAer13/mplug-owl2-llama2-7b"
        model_name = get_model_name_from_path(model_path)
        tokenizer, lmm_model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            load_8bit=False,
            load_4bit=False,
            device=device,
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        get_desc_function = partial(
            get_desc_mPLUG, image_processor, lmm_model, tokenizer, prompt
        )
        encode_image_function = encode_image_os
    elif args.model == "gpt4v":
        get_desc_function = partial(get_desc_gpt4v, prompt=prompt)
        encode_image_function = encode_image_gpt4v

    output_folder = os.path.join(args.dataset, f"results_{args.model}")
    os.makedirs(output_folder, exist_ok=True)

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
