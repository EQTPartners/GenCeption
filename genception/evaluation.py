import os
import json
import pickle
import numpy as np
import argparse
from genception.utils import find_files


def read_all_pkl(folder_path: str) -> dict:
    """
    Read all the pickle files in the given folder path
    
    Args:
    folder_path: str: The path to the folder
    
    Returns:
    dict: The dictionary containing the file path as key and the pickle file content as value
    """
    result_dict = dict()
    file_list = find_files(folder_path, {".pkl"})
    for file_path in file_list:
        with open(file_path, "rb") as file:
            result_dict[file_path] = pickle.load(file)
    return result_dict


def integrated_decay_area(scores: list[float]) -> float:
    """
    Calculate the Integrated Decay Area (IDA) for the given scores

    Args:
    scores: list[float]: The list of scores

    Returns:
    float: The IDA score
    """
    total_area = 0

    for i, score in enumerate(scores):
        total_area += (i + 1) * score

    max_possible_area = sum(range(1, len(scores) + 1))
    ida = total_area / max_possible_area if max_possible_area else 0
    return ida


def gc_score(folder_path: str, n_iter: int = None) -> tuple[float, list[float]]:
    """
    Calculate the GC@T score for the given folder path

    Args:
    folder_path: str: The path to the folder
    n_iter: int: The number of iterations to consider for GC@T score

    Returns:
    tuple[float, list[float]]: The GC@T score and the list of GC scores for each file
    """
    test_data = read_all_pkl(folder_path)
    all_gc_scores = []
    for _, value in test_data.items():
        sim_score = value["cosine_similarities"][1:]
        if n_iter is None:
            _gc = integrated_decay_area(sim_score)
        else:
            if len(value["cosine_similarities"]) >= n_iter:
                _gc = integrated_decay_area(sim_score[:n_iter])
            else:
                continue
        all_gc_scores.append(_gc)
    return np.mean(all_gc_scores), all_gc_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to the folder containing the pickle files",
        required=True,
    )
    parser.add_argument(
        "--t",
        type=int,
        help="Number of iterations to consider for GC@T score",
        required=True,
    )
    args = parser.parse_args()

    # calculate GC@T score and save in results directory
    gc, all_gc_scores = gc_score(args.results_path, args.t)
    result = {
        "GC Score": gc,
        "All GC Scores": all_gc_scores,
    }
    results_path = os.path.join(args.results_path, f"GC@{str(args.t)}.json")
    with open(results_path, "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    main()
