from collections import defaultdict
import pickle

import torch
import numpy as np

combined = defaultdict(dict)
losses =[]
test_accs = []
noisy_test_accs = []
masked_test_accs = []

for num_layers in range(2, 5):
    for num_recurrence in range(3):
        num_channels_dict = {}
        for num_channels in [8, 16, 24]:
            with open(
                f"./model_num_layers={num_layers}_num_recurrence={num_recurrence}_num_channels={num_channels}.pkl",
                "rb",
            ) as f:
                data = pickle.load(f)
            print(
                f"num_layers={num_layers}, num_recurrence={num_recurrence}, num_channels={num_channels}, test_acc={data['test_acc']}, test_noisy_acc={data['test_noisy_acc']}, test_masked_acc={data['test_masked_acc']}"
            )
            num_channels_dict[f"num_channels={num_channels}"] = data
            
            losses.append(data["loss"])
            test_accs.append(data["test_acc"])
            noisy_test_accs.append(data["test_noisy_acc"])
            masked_test_accs.append(data["test_masked_acc"])
        combined[f"num_layers={num_layers}"][f"num_recurrence={num_recurrence}"] = num_channels_dict

losses = np.array(losses).reshape(3, 3, 3, -1)
test_accs = np.array(test_accs).reshape(3, 3, 3)
noisy_test_accs = np.array(noisy_test_accs).reshape(3, 3, 3)
masked_test_accs = np.array(masked_test_accs).reshape(3, 3, 3)

combined_list = {'losses': losses, 'test_accs': test_accs, 'noisy_test_accs': noisy_test_accs, 'masked_test_accs': masked_test_accs}
with open(f"./combined.pkl", "wb") as f:
    pickle.dump(combined, f)

with open(f"./combined_list.pkl", "wb") as f:
    pickle.dump(combined_list, f)