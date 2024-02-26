# run experiment with gpt4v on examples dataset
python genception/experiment.py --model gpt4v --dataset datasets/examples


# Calculate GC@T evaluation metric
python genception/evaluation.py --results_path datasets/examples/results_gpt4v --t 1
python genception/evaluation.py --results_path datasets/examples/results_gpt4v --t 3
python genception/evaluation.py --results_path datasets/examples/results_gpt4v --t 5
