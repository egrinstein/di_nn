import torch

from torch.nn import Module

DEFAULT_STFT_CONFIG = {"n_fft": 1024, "use_onesided_fft":True}


class StftArray(Module):
    def __init__(self, model_config):

        super().__init__()

        self.n_fft = model_config["n_fft"]
        self.onesided = model_config["use_onesided_fft"]
        self.is_complex = True

    def forward(self, X):
        "Expected input has shape (batch_size, n_arrays, time_steps)"

        result = []
        n_arrays = X.shape[1]

        for i in range(n_arrays):
            x = X[:, i, :]
            stft_output = torch.stft(x, self.n_fft, onesided=self.onesided,
                                     return_complex=True, normalized=True)
            result.append(
                stft_output[:, 1:, :]
            ) # Ignore frequency 0
        
        result = torch.stack(result, dim=1) # Should transpose?
        return result


class DecoupledStftArray(StftArray):
    "Stft where the real and imaginary channels are modeled as separate channels"
    def __init__(self, model_config):
        super().__init__(model_config)
        self.is_complex = False

    def forward(self, X):

        stft = super().forward(X)

        # stft.real.shape = (batch_size, num_mics, num_channels, time_steps)
        result = torch.cat((stft.real, stft.imag), dim=2)   
        
        return result


FEATURE_NAME_TO_CLASS_MAP = {
    "stft": StftArray,
    "decoupled_stft": DecoupledStftArray
}