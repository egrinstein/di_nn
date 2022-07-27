from .distributed_ssl_dataset import DistributedSSLDataset

import argparse
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm


def compute_dataset_microphone_distance(dataset_1_path, dataset_2_path):

    print("Loading dataset...")
    dataset_1 = DistributedSSLDataset(dataset_1_path, stack_parameters=False)
    dataset_2 = DistributedSSLDataset(dataset_2_path, stack_parameters=False)

    mic_coords_1 = torch.stack([d[0]["mic_coordinates"] for d in dataset_1])
    mic_coords_2 = torch.stack([d[0]["mic_coordinates"] for d in dataset_2])
    print("Dataset loaded.")

    print("Computing minimum distances...")
    min_dists = []
    for mic_coords in tqdm(mic_coords_2):
        mic_coords = mic_coords.repeat(mic_coords_1.shape[0], 1, 1)

        dists = torch.sum((mic_coords - mic_coords_1)**2, dim=-1)
        summed_dists = torch.sum(dists, axis=-1)
        min_dists.append(summed_dists.min())
    
    min_dists = torch.stack(min_dists).numpy()
    return min_dists


def _save_histogram(min_dists):
    print("Min distance:", min_dists.min())
    fig=plt.figure()
    plt.title("Train vs test dataset minimum cum. microphone distance")
    plt.xlabel("Minimum distance from training samples (m)")
    plt.ylabel("Frequency (samples)")
    plt.ion()
    plt.hist(min_dists, bins=50)
    plt.ioff()
    plt.savefig("hist.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_1_path")
    parser.add_argument("dataset_2_path")
    args = parser.parse_args()

    min_dists = compute_dataset_microphone_distance(args.dataset_1_path, args.dataset_2_path)
    _save_histogram(min_dists)
