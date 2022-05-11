import hydra
import torch

from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from complex_neural_source_localization.datasets import create_torch_dataloader
from pysoundloc.pysoundloc.least_squares_localization import least_squares_sound_localization

SR = 16000
GRID_SEARCH_RESOLUTION = 0.02


def test_batch(batch):
    x, y = batch
    signals = x["signal"].numpy()
    mic_coordinates = x["mic_coordinates"].numpy()
    room_dims = x["room_dims"].numpy()

    n_signals = signals.shape[0]

    results = []

    for i in range(n_signals):
        coordinates, grid = least_squares_sound_localization(
                                signals[i], SR, mic_coordinates[i], room_dims[i][0],
                                resolution_in_meters=GRID_SEARCH_RESOLUTION)
        results.append(coordinates)
    results = torch.Tensor(results)

    mean_absolute_error = torch.mean(
        torch.abs(results - y["source_coordinates"])
    )

    return {
        "errors": mean_absolute_error,
        "results": results
    }


@hydra.main(config_path="config", config_name="config")
def test(config: DictConfig):
    """Runs the training procedure using Pytorch lightning
    And tests the model with the best validation score against the test dataset. 

    Args:
        config (DictConfig): Configuration automatically loaded by Hydra.
                                        See the config/ directory for the configuration
    """

    dataset_test = create_torch_dataloader(config, "test", stack_parameters=False)
    
    results = Parallel(n_jobs=config["training"]["n_workers"])(
        delayed(test_batch)(d)
        for d in tqdm(dataset_test))
    
    errors = [result["errors"] for result in results]
    errors = torch.stack(errors)
    error = torch.mean(errors)
    print("Mean dataset error:", error)


if __name__ == "__main__":
    test()