import os


def find_files(folder_path: str, file_extensions: dict) -> list[str]:
    """
    Find all files with the given extensions in the given folder path

    Args:
    folder_path: str: The path to the folder
    file_extensions: dict: The file extensions to look for

    Returns:
    list[str]: The list of file paths
    """
    file_paths = []

    for file in os.listdir(folder_path):
        if (
            os.path.isfile(os.path.join(folder_path, file))
            and os.path.splitext(file)[1].lower() in file_extensions
        ):
            absolute_path = os.path.abspath(os.path.join(folder_path, file))
            file_paths.append(absolute_path)

    return file_paths


def find_image_files(folder_path: str) -> list[str]:
    image_extensions = {".jpg", ".png"}
    return find_files(folder_path, image_extensions)