import os
import io
import time
import base64
import logging
import pickle
import requests
from nltk.tokenize import word_tokenize
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# VIT model
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# The prompt used to generated description of an image
prompt = (
    "Please write a clear, precise, detailed, and concise description of all elements in the image. "
    "Focus on accurately depicting various aspects, including but not limited to the colors, shapes, "
    "positions, styles, texts and the relationships between different objects and subjects in the image. "
    "Your description should be thorough enough to guide a professional in recreating this image solely "
    "based on your textual representation. Remember, only include descriptive texts that directly pertain "
    "to the contents of the image. You must complete the description using less than 500 words."
)


def encode_image_base64(image_path: str):
    with Image.open(image_path) as img:
        # Convert the image to JPEG format in memory
        with io.BytesIO() as buffer:
            img.save(buffer, format="JPEG")
            jpeg_img_bytes = buffer.getvalue()
            return base64.b64encode(jpeg_img_bytes).decode("utf-8")


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
        # Convert content to a BytesIO stream to be read by PIL
        image_stream = io.BytesIO(response.content)
        image = Image.open(image_stream)

        # Ensure the filename ends with .jpg extension
        _base, _extension = os.path.splitext(filename)
        if not _extension.lower() == '.jpg':
            filename = _base + '.jpg'

        # Convert to JPEG and save
        image.convert('RGB').save(filename, "JPEG")

        return filename
    else:
        logging.warning(
            f"Failed to download image. Status code: {response.status_code}"
        )


def count_words(text):
    words = word_tokenize(text)
    return len(words)


def encode_image_os(image_path: str):
    image = Image.open(image_path).convert("RGB")
    return image


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
    return save_image_from_url(response.data[0].url, new_image_filename)


def test_sample(
    seed_image: str,
    n_iteration: int,
    output_folder: str,
    get_desc_function: callable,
    encode_image_function: callable,
    sleep_between_iterations: int=0,
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
    iter_start = 0

    current_image_path = seed_image
    current_image_name = os.path.basename(current_image_path)
    file_name, file_extension = current_image_name.split(".")
    logging.debug(f"Image: {current_image_path}")
    
    # Handeling cases where GenCeption has been run (even partially) for the specified seed image.
    pkl_file = os.path.join(output_folder, f"{file_name}_result.pkl")
    if os.path.exists(pkl_file):
        logging.info("Results already exist, checking completeness ...")
        with open(pkl_file, "rb") as file:
            _pkl_data = pickle.load(file)
            _n_iter_run = len(_pkl_data["images"])
            if _n_iter_run >= n_iteration:
                return None
            else:
                iter_start = _n_iter_run
                current_image_path = os.path.join(output_folder, f"{file_name}_{_n_iter_run-1}.jpg")
                list_of_desc = _pkl_data["descriptions"]
                list_of_image = _pkl_data["images"]
                list_of_image_embedding = _pkl_data["image_embeddings"]
                list_of_cos_sim = _pkl_data["cosine_similarities"]
                logging.info(f"{_n_iter_run} has run, starting from {current_image_path}.")

    for i in range(iter_start, n_iteration):
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

        # Sleep to cope with rate control for some models
        if sleep_between_iterations > 0:
            time.sleep(sleep_between_iterations)

    return None