import argparse
from collections import defaultdict
import pickle

import torch
import numpy as np

parser = argparse.ArgumentParser(description="Fit different DCAs.")
parser.add_argument("--experiment", type=int, default=0, help="Experiment number")
args = parser.parse_args()

seeds = [0, 10, 100, 1000, 10000, 1000000]
seed = seeds[args.experiment]
for dataset in ["cifar10"]:
    combined = defaultdict(dict)
    losses = []
    test_accs = []
    noisy_test_accs = []
    masked_test_accs = []
    left_masked_test_accs = []
    gaussian_blur_test_accs = []
    for num_layers in range(2, 5):
        for num_recurrence in range(3):
            num_channels_dict = {}
            for num_channels in [8, 16, 24]:
                with open(
                    f"./models/{dataset}_seed_{seed}_model_bn_num_layers={num_layers}_num_recurrence={num_recurrence}_num_channels={num_channels}.pkl",
                    "rb",
                ) as f:
                    data = pickle.load(f)
                print(
                    f"{dataset}_seed_{seed}_num_layers={num_layers}, num_recurrence={num_recurrence}, num_channels={num_channels}, test_acc={data['test_acc']}, test_noisy_acc={data['test_noisy_acc']}, test_masked_acc={data['test_masked_acc']}"
                )
                num_channels_dict[f"num_channels={num_channels}"] = data

                losses.append(data["loss"])
                test_accs.append(data["test_acc"])
                noisy_test_accs.append(data["test_noisy_acc"])
                masked_test_accs.append(data["test_masked_acc"])
                left_masked_test_accs.append(data["test_left_masked_acc"])
                gaussian_blur_test_accs.append(data["test_gaussian_blur_acc"])

            combined[f"num_layers={num_layers}"][
                f"num_recurrence={num_recurrence}"
            ] = num_channels_dict

    losses = np.array(losses).reshape(3, 3, 3, -1)
    test_accs = np.array(test_accs).reshape(3, 3, 3)
    noisy_test_accs = np.array(noisy_test_accs).reshape(3, 3, 3)
    masked_test_accs = np.array(masked_test_accs).reshape(3, 3, 3)
    left_masked_test_accs = np.array(left_masked_test_accs).reshape(3, 3, 3)
    gaussian_blur_test_accs = np.array(gaussian_blur_test_accs).reshape(3, 3, 3)

    combined_list = {
        "losses": losses,
        "test_accs": test_accs,
        "test_noisy_accs": noisy_test_accs,
        "test_masked_accs": masked_test_accs,
        "test_left_masked_accs": left_masked_test_accs,
        "test_gaussian_blur_accs": gaussian_blur_test_accs,
    }
    print(combined_list.keys())
    with open(f"./results/{dataset}_bn_seed={seed}_combined.pkl", "wb") as f:
        pickle.dump(combined, f)

    with open(f"./results/{dataset}_bn_seed={seed}_combined_list.pkl", "wb") as f:
        pickle.dump(combined_list, f)
