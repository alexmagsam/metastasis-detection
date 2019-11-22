import argparse
import torch
import data
import networks
import numpy as np
import csv
import os
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-data_path', required=True, help="Path to test HDF5 files.")
parser.add_argument('-results_path', default='results/', help='Folder to save results file in')
parser.add_argument('-saved_model', required=True, help="Path to trained model file")
parser.add_argument('-batch_size', default=32, type=int, help='Batch size')
parser.add_argument('-device', default=0, type=int, help='CUDA device')
args = parser.parse_args()
print(args)


# Device
device = "cuda:{}".format(args.device) if torch.cuda.is_available() else 'cpu'

# Test dataset
test_dataset = data.PatchCamelyon(args.data_path, mode='test')

# Create the model and load the saved model file
model = networks.CamelyonClassifier()
model.load_state_dict(torch.load(args.saved_model))
model.to(device)
model.eval()


def test():

    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'specificity': []}
    labels_npy = []
    predictions_npy = []
    for idx in range(len(test_dataset)):

        # Load data
        sample = test_dataset[idx]
        images = sample['images'].to(device)
        labels = sample['labels'].to(device)

        # Predict on batch of images
        predictions = model(images)

        #
        labels_npy.append(labels.cpu().detach().numpy().squeeze())
        predictions_npy.append(predictions.cpu().detach().numpy().squeeze())

        # Track batch results
        m = utils.metrics(predictions, labels)
        for key in m.keys():
            results[key].append(m[key])

    # Plot the confusion matrix
    labels_npy = np.array(labels_npy).reshape(-1)
    predictions_npy = (np.array(predictions_npy) > 0.5).astype(np.float32).reshape(-1)
    np.save('labels_test.npy', labels_npy)
    np.save('predictions_test.npy', predictions_npy)

    # Get the average over all batches
    for key in results.keys():
        results[key] = np.mean(results[key])

    w = csv.writer(open(os.path.join(args.results_path, "results.csv"), "w"))
    for key, val in results.items():
        w.writerow([key, val])


if __name__ == '__main__':
    test()
