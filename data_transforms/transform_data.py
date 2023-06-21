"""
Contains Pytorch models to apply data transformations.
"""
import copy
from sys import exit
from functools import lru_cache

import neurokit2 as nk
import numpy as np
import scipy
from scipy.fft import fft, ifft
from scipy.signal import iirnotch, iirpeak, filtfilt
from scipy.interpolate import CubicSpline


class DataTransformer:
    def __init__(self, data_transform_names, data_transform_args):
        """
        :param data_transform_names: Names of data transformations to randomly sample from
        :param data_transform_args: Arguments associated with each of the transformations
        """
        self.transform_names = data_transform_names
        self.transform_args = data_transform_args
        self.trfs = self.get_transforms()
        # n_transforms: number of transforms to apply to single sample
        # stochastic choice: if true, randomly sample from list of provided transforms.
        # if false, apply in sequential order
        self.n_transforms = self.transform_args["n_transforms"]
        self.stochastic_choice = self.transform_args["stochastic_choice"]

    def get_transforms(self):
        trfs = []
        for trf_name in self.transform_names:
            if trf_name == "GaussianNoiseDeterministic":
                assert "GaussianNoiseDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for Gaussian Noise Deterministic transform"
                trfs.append(GaussianNoiseDeterministic(**self.transform_args["GaussianNoiseDeterministic"]))
            elif trf_name == "GaussianNoiseStochastic":
                assert "GaussianNoiseStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for Gaussian Noise Stochastic transform"
                trfs.append(GaussianNoiseStochastic(**self.transform_args["GaussianNoiseStochastic"]))

            elif trf_name == "BandstopFilterDeterministic":
                assert "BandstopFilterDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for Bandstop filter deterministic transform"
                trfs.append(BandstopFilterDeterministic(**self.transform_args["BandstopFilterDeterministic"]))
            elif trf_name == "BandstopFilterStochastic":
                assert "BandstopFilterStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for Bandstop filter transform stochastic"
                trfs.append(BandstopFilterStochastic(**self.transform_args["BandstopFilterStochastic"]))

            elif trf_name == "BandpassFilterDeterministic":
                assert "BandpassFilterDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for Bandpass filter deterministic transform"
                trfs.append(BandpassFilterDeterministic(**self.transform_args["BandpassFilterDeterministic"]))
            elif trf_name == "BandpassFilterStochastic":
                assert "BandpassFilterStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for Bandpass filter transform stochastic"
                trfs.append(BandpassFilterStochastic(**self.transform_args["BandpassFilterStochastic"]))

            elif trf_name == "TemporalCutoutDeterministic":
                assert "TemporalCutoutDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for temporal cutout transform deterministic"
                trfs.append(TemporalCutoutDeterministic(**self.transform_args["TemporalCutoutDeterministic"]))
            elif trf_name == "TemporalCutoutStochastic":
                assert "TemporalCutoutStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for temporal cutout transform stochastic"
                trfs.append(TemporalCutoutStochastic(**self.transform_args["TemporalCutoutStochastic"]))

            elif trf_name == "TimeShiftDeterministic":
                assert "TimeShiftDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for time shift deterministic transform"
                trfs.append(TimeShiftDeterministic(**self.transform_args["TimeShiftDeterministic"]))
            elif trf_name == "TimeShiftStochastic":
                assert "TimeShiftStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for time shift stochastic transform"
                trfs.append(TimeShiftStochastic(**self.transform_args["TimeShiftStochastic"]))

            elif trf_name == "PermuteDeterministic":
                assert "PermuteDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for permute deterministic transform"
                trfs.append(PermuteDeterministic(**self.transform_args["PermuteDeterministic"]))
            elif trf_name == "PermuteStochastic":
                assert "PermuteStochastic" in self.transform_args.keys(), \
                    "need to provide arguents for permute stochastic transform"
                trfs.append(PermuteStochastic(**self.transform_args["PermuteStochastic"]))

            elif trf_name == "TimeWarpingDeterministic":
                assert "TimeWarpingDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for time warping deterministic transform"
                trfs.append(TimeWarpingDeterministic(**self.transform_args["TimeWarpingDeterministic"]))
            elif trf_name == "TimeWarpingStochastic":
                assert "TimeWarpingStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for time warping stochastic transform"
                trfs.append(TimeWarpingStochastic(**self.transform_args["TimeWarpingStochastic"]))

            elif trf_name == "Flip":  # no args for flip
                trfs.append(Flip())

            elif trf_name == "ExtractPhasic":
                assert "ExtractPhasic" in self.transform_args.keys(), \
                    "need to provide arguments for extract phasic transform"
                trfs.append(ExtractPhasic(**self.transform_args["ExtractPhasic"]))
            elif trf_name == "ExtractTonic":
                assert "ExtractTonic" in self.transform_args.keys(), \
                    "need to provide arguments for extract tonic transform"
                trfs.append(ExtractTonic(**self.transform_args["ExtractTonic"]))

            elif trf_name == "LowPassFilterDeterministic":
                assert "LowPassFilterDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for low pass filter deterministic transform"
                trfs.append(LowPassFilterDeterministic(**self.transform_args["LowPassFilterDeterministic"]))
            elif trf_name == "LowPassFilterStochastic":
                assert "LowPassFilterStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for low pass filter stochastic transform"
                trfs.append(LowPassFilterStochastic(**self.transform_args["LowPassFilterStochastic"]))

            elif trf_name == "HighPassFilterDeterministic":
                assert "HighPassFilterDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for high pass filter deterministic transform"
                trfs.append(HighPassFilterDeterministic(**self.transform_args["HighPassFilterDeterministic"]))
            elif trf_name == "HighPassFilterStochastic":
                assert "HighPassFilterStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for high pass filter stochastic transform"
                trfs.append(HighPassFilterStochastic(**self.transform_args["HighPassFilterStochastic"]))

            elif trf_name == "HighFrequencyNoiseDeterministic":
                assert "HighFrequencyNoiseDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for high frequency noise transform deterministic"
                trfs.append(HighFrequencyNoiseDeterministic(**self.transform_args["HighFrequencyNoiseDeterministic"]))
            elif trf_name == "HighFrequencyNoiseStochastic":
                assert "HighFrequencyNoiseStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for high frequency noise transform stochastic"
                trfs.append(HighFrequencyNoiseStochastic(**self.transform_args["HighFrequencyNoiseStochastic"]))

            elif trf_name == "LooseSensorArtifactDeterministic":
                assert "LooseSensorArtifactDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for loose sensor artifact transform deterministic"
                trfs.append(LooseSensorArtifactDeterministic(**self.transform_args["LooseSensorArtifactDeterministic"]))
            elif trf_name == "LooseSensorArtifactStochastic":
                assert "LooseSensorArtifactStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for loose sensor artifact transform stochastic"
                trfs.append(LooseSensorArtifactStochastic(**self.transform_args["LooseSensorArtifactStochastic"]))
            
            elif trf_name == "JumpArtifactDeterministic":
                assert "JumpArtifactDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for jump artifact transform deterministic"
                trfs.append(JumpArtifactDeterministic(**self.transform_args["JumpArtifactDeterministic"]))
            elif trf_name == "JumpArtifactStochastic":
                assert "JumpArtifactStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for jump artifact transform stochastic"
                trfs.append(JumpArtifactStochastic(**self.transform_args["JumpArtifactStochastic"]))

            elif trf_name == "ConstantAmplitudeScalingDeterministic":
                assert "ConstantAmplitudeScalingDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for constant scaling transform deterministic"
                trfs.append(ConstantAmplitudeScalingDeterministic(**self.transform_args["ConstantAmplitudeScalingDeterministic"]))
            elif trf_name == "ConstantAmplitudeScalingStochastic":
                assert "ConstantAmplitudeScalingStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for constant scaling transform stochastic"
                trfs.append(ConstantAmplitudeScalingStochastic(**self.transform_args["ConstantAmplitudeScalingStochastic"]))

            elif trf_name == "AmplitudeWarpingDeterministic":
                assert "AmplitudeWarpingDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for magnitude warping transform deterministic"
                trfs.append(AmplitudeWarpingDeterministic(**self.transform_args["AmplitudeWarpingDeterministic"]))
            elif trf_name == "AmplitudeWarpingStochastic":
                assert "AmplitudeWarpingStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for magnitude warping transform stochastic"
                trfs.append(AmplitudeWarpingStochastic(**self.transform_args["AmplitudeWarpingStochastic"]))

            elif trf_name == "TonicConstantAmplitudeScalingDeterministic":
                assert "TonicConstantAmplitudeScalingDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for TONIC constant scaling transform deterministic"
                trfs.append(TonicConstantAmplitudeScalingDeterministic(**self.transform_args["TonicConstantAmplitudeScalingDeterministic"]))
            elif trf_name == "TonicConstantAmplitudeScalingStochastic":
                assert "TonicConstantAmplitudeScalingStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for TONIC constant scaling transform stochastic"
                trfs.append(TonicConstantAmplitudeScalingStochastic(**self.transform_args["TonicConstantAmplitudeScalingStochastic"]))

            elif trf_name == "TonicAmplitudeWarpingDeterministic":
                assert "TonicAmplitudeWarpingDeterministic" in self.transform_args.keys(), \
                    "need to provide arguments for TONIC magnitude warping transform deterministic"
                trfs.append(TonicAmplitudeWarpingDeterministic(**self.transform_args["TonicAmplitudeWarpingDeterministic"]))
            elif trf_name == "TonicAmplitudeWarpingStochastic":
                assert "TonicAmplitudeWarpingStochastic" in self.transform_args.keys(), \
                    "need to provide arguments for TONIC magnitude warping transform stochastic"
                trfs.append(TonicAmplitudeWarpingStochastic(**self.transform_args["TonicAmplitudeWarpingStochastic"]))

            elif trf_name == "FlipWrist":
                trfs.append(FlipWrist())
            elif trf_name == "ScalePhasic":
                assert "ScalePhasic" in self.transform_args.keys(), \
                    "need to provide arguments for scale phasic transform"
                trfs.append(ScalePhasic(**self.transform_args["ScalePhasic"]))
            elif trf_name == "DCShift":
                assert "DCShift" in self.transform_args.keys(), \
                    "need to provide arguments for DC shift transform"
                trfs.append(DCShift(**self.transform_args["DCShift"]))
            elif trf_name == "Identity":
                trfs.append(Identity())
            else:
                print(f"Unrecognized data transform name {trf_name}")
                print("Exiting...")
                exit(1)
        return trfs

    def __call__(self, sample_dict):
        """
        :param sample_dict: sample_dict containing 'x' to apply transformation to
                            also contains left and right buffers (for shift transform)
        :return: transformed sample
        """
        # reset 'x' to 'x_base' (to undo transforms from previous calls to data transformer)
        sample_dict['x'] = copy.deepcopy(sample_dict['x_base'])
        for i in range(self.n_transforms):
            if self.stochastic_choice:
                # randomly sample transformation to apply to sample
                trf_idx = np.random.choice(len(self.trfs))
                trf_fn = self.trfs[trf_idx]
            else:  # apply transforms sequentially
                trf_fn = self.trfs[i]
            # apply transform
            # need .copy() to fix negative stride issue
            # see https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
            sample_dict['x'] = trf_fn(sample_dict).copy()
        return sample_dict['x']


class GaussianNoiseDeterministic:
    def __init__(self, sigma_scale=0.1):
        """
        :param sigma_scale: factor to use in computing sigma parameter for noise distribution
            sigma = mean(abs(diff between signal & mean))) * sigma_scale
        """
        self.sigma_scale = sigma_scale

    def __call__(self, sample_dict):
        x = sample_dict['x']
        mean_power_diff = np.mean(np.abs(x - np.mean(x)))
        noise_sigma = mean_power_diff * self.sigma_scale
        noise = np.random.normal(scale=noise_sigma, size=len(x))
        return x + noise


class GaussianNoiseStochastic:
    def __init__(self, sigma_scale_min=0.0, sigma_scale_max=0.5):
        """
        :param sigma_scale_min: min factor to use in computing sigma parameter for noise distribution
        :param sigma_scale_max: max factor to use in computing sigma parameter for noise distribution
            sample sigma_scale uniformly in [sigma_scale_min, sigma_scale_max)
            sigma = mean(abs(diff between signal & mean))) * sigma_scale
        """
        self.sigma_scale_min = sigma_scale_min
        self.sigma_scale_max = sigma_scale_max

    def __call__(self, sample_dict):
        x = sample_dict['x']
        # sample sigma scale
        sigma_scale = np.random.uniform(self.sigma_scale_min, self.sigma_scale_max)
        mean_power_diff = np.mean(np.abs(x - np.mean(x)))
        noise_sigma = mean_power_diff * sigma_scale
        noise = np.random.normal(scale=noise_sigma, size=len(x))
        return x + noise


class Identity:
    def __call__(self, sample_dict):
        return sample_dict['x']


class LowPassFilterDeterministic:
    def __init__(self, data_freq=4, highcut_hz=0.05):
        """
        Apply low pass filter to remove frequency bands >= highcut_hz
        :param data_freq: frequency of data to apply filter to (e.g., 4Hz for EDA)
        :param highcut_hz: lower bound on frequency bands to remove
        """
        self.data_freq = data_freq
        self.highcut_hz = highcut_hz
        self.b, self.a = scipy.signal.butter(4, [highcut_hz], btype="lowpass", output="ba", fs=data_freq)

    def __call__(self, sample_dict):
        x = sample_dict['x']
        segment_filtered = scipy.signal.filtfilt(self.b, self.a, x)
        return segment_filtered


@lru_cache(maxsize=None)
def memoize_scipy_lowpass_butter_design_ba(order, highcut_hz, fs):
    b, a = scipy.signal.butter(order, [highcut_hz], btype="lowpass", output="ba", fs=fs)
    return b, a


class LowPassFilterStochastic:
    def __init__(self, data_freq=4, highcut_hz_min=0.01, highcut_hz_max=1, n_steps=1000):
        """
        Apply low pass filter to remove frequency bands >= highcut_hz
        :param data_freq: frequency of data to apply filter to (e.g., 4Hz for EDA)
        :param highcut_hz_min: min cutoff freq to use
        :param highcut_hz_max: max cutoff freq to use (sample from [min, max))
        """
        self.data_freq = data_freq
        self.highcut_hz_min = highcut_hz_min
        self.highcut_hz_max = highcut_hz_max
        self.n_steps = n_steps

    def __call__(self, sample_dict):
        x = sample_dict['x']
        # discretise the space so we get some speed benefits 
        highcut_hz = np.random.choice(np.linspace(self.highcut_hz_min, self.highcut_hz_max, self.n_steps))
        b, a = memoize_scipy_lowpass_butter_design_ba(
            order=4, highcut_hz=highcut_hz, fs=self.data_freq,
        )
        segment_filtered = scipy.signal.filtfilt(b, a, x)
        return segment_filtered


class HighPassFilterDeterministic:
    def __init__(self, data_freq=4, lowcut_hz=0.05):
        """
        Apply high pass filter to remove frequency bands <= lowcut_hz
        :param data_freq: frequency of data to apply filter to (e.g., 4Hz for EDA)
        :param lowcut_hz: upper bound on frequency bands to remove
        """
        self.data_freq = data_freq
        self.lowcut_hz = lowcut_hz
        self.b, self.a = scipy.signal.butter(4, [lowcut_hz], btype="highpass", output="ba", fs=data_freq)

    def __call__(self, sample_dict):
        x = sample_dict['x']
        segment_filtered = scipy.signal.filtfilt(self.b, self.a, x)
        return segment_filtered
    

@lru_cache(maxsize=None)
def memoize_scipy_highpass_butter_design_ba(order, lowcut_hz, fs):
    b, a = scipy.signal.butter(order, [lowcut_hz], btype="highpass", output="ba", fs=fs)
    return b, a


class HighPassFilterStochastic:
    def __init__(self, data_freq=4, lowcut_hz_min=0.01, lowcut_hz_max=1, n_steps=1000):
        """
        Apply high pass filter to remove frequency bands <= lowcut_hz
        :param data_freq: frequency of data to apply filter to (e.g., 4Hz for EDA)
        :param lowcut_hz: upper bound on frequency bands to remove
        """
        self.data_freq = data_freq
        self.lowcut_hz_min = lowcut_hz_min
        self.lowcut_hz_max = lowcut_hz_max
        self.n_steps = n_steps

    def __call__(self, sample_dict):
        x = sample_dict['x']
        lowcut_hz = np.random.choice(np.linspace(self.lowcut_hz_min, self.lowcut_hz_max, self.n_steps))
        b, a = memoize_scipy_highpass_butter_design_ba(
            order=4, lowcut_hz=lowcut_hz, fs=self.data_freq,
        )
        segment_filtered = scipy.signal.filtfilt(b, a, x)
        return segment_filtered


class HighFrequencyNoiseDeterministic:
    def __init__(self, sigma_scale=0.1, freq_bin_start_idx=60, freq_bin_stop_idx=120):
        self.sigma_scale = sigma_scale
        self.freq_bin_start_idx = freq_bin_start_idx
        self.req_bin_stop_idx = freq_bin_stop_idx
        self.freq_bin_idxs = np.arange(freq_bin_start_idx, freq_bin_stop_idx)

    def __call__(self, sample_dict):
        x = sample_dict['x']
        x_fft = fft(x)
        mean_fft_val = np.mean(np.abs(x_fft))
        sigma = self.sigma_scale * mean_fft_val
        noise = np.random.normal(scale=sigma, size=len(self.freq_bin_idxs))
        x_fft[self.freq_bin_idxs] += noise
        # get the corresponding negative bins
        neg_end_idx = len(x) + 1 - self.freq_bin_start_idx
        neg_start_idx = neg_end_idx - len(self.freq_bin_idxs)
        x_fft[neg_start_idx:neg_end_idx] += np.flip(noise)
        x_ifft = np.abs(ifft(x_fft))
        return x_ifft


class HighFrequencyNoiseStochastic:
    def __init__(self, sigma_scale_min=0.0, sigma_scale_max=1.0, freq_bin_start_idx=60, freq_bin_stop_idx=120):
        self.sigma_scale_min = sigma_scale_min
        self.sigma_scale_max = sigma_scale_max
        self.freq_bin_start_idx = freq_bin_start_idx
        self.req_bin_stop_idx = freq_bin_stop_idx
        self.freq_bin_idxs = np.arange(freq_bin_start_idx, freq_bin_stop_idx)

    def __call__(self, sample_dict):
        x = sample_dict['x']
        x_fft = fft(x)
        mean_fft_val = np.mean(np.abs(x_fft))
        # sample sigma scale
        sigma_scale = np.random.uniform(self.sigma_scale_min, self.sigma_scale_max)
        sigma = sigma_scale * mean_fft_val
        noise = np.random.normal(scale=sigma, size=len(self.freq_bin_idxs))
        x_fft[self.freq_bin_idxs] += noise
        # get the corresponding negative bins
        neg_end_idx = len(x) + 1 - self.freq_bin_start_idx
        neg_start_idx = neg_end_idx - len(self.freq_bin_idxs)
        x_fft[neg_start_idx:neg_end_idx] += np.flip(noise)
        x_ifft = np.abs(ifft(x_fft))
        return x_ifft


class BandstopFilterDeterministic:
    def __init__(self, data_freq=4, remove_freq=0.25, Q=0.707):
        """
        Selects frequency band to remove.
        See https://stackoverflow.com/questions/54320638/how-to-create-a-bandstop-filter-in-python
        :param data_freq: frequency of data to apply filter to (e.g., 4Hz for EDA)
        :param remove_freq: frequency band to remove
        :param Q: "quality factor" Q = remove_freq / width of filter
        see - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
        and https://en.wikipedia.org/wiki/Q_factor
        """
        self.data_freq = data_freq
        self.remove_freq = remove_freq
        self.Q = Q
        self.b, self.a = iirnotch(self.remove_freq, self.Q, fs=self.data_freq)

    def __call__(self, sample_dict):
        x = sample_dict['x']
        return filtfilt(self.b, self.a, x)


class BandstopFilterStochastic:
    def __init__(self, data_freq=4, remove_freq_min=0.01, remove_freq_max=1.0, Q=0.707):
        """
        Randomly selects frequency band to remove.
        See https://stackoverflow.com/questions/54320638/how-to-create-a-bandstop-filter-in-python
        :param data_freq: frequency of data to apply filter to (e.g., 4Hz for EDA)
        :param remove_freq_min: minimum frequency band to remove
        :param remove_freq_max: maximum frequency band to remove (sample remove_freq from uniform [min, max))
        :param Q: "quality factor" Q = remove_freq / width of filter
        see - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
        and https://en.wikipedia.org/wiki/Q_factor
        """
        self.data_freq = data_freq
        self.remove_feq_min = remove_freq_min
        self.remove_freq_max = remove_freq_max
        self.Q = Q

    def __call__(self, sample_dict):
        x = sample_dict['x']
        # sample frequency band to remove
        remove_freq = np.random.uniform(self.remove_feq_min, self.remove_freq_max)
        b, a = iirnotch(remove_freq, self.Q, fs=self.data_freq)
        return filtfilt(b, a, x)


class BandpassFilterDeterministic:
    def __init__(self, data_freq=4, keep_freq=0.25, Q=0.707):
        """
        Select frequency band to keep
        :param keep_freq: central frequency to keep
        :param Q: "quality factor" Q = width of filter
        """
        self.data_freq = data_freq
        self.keep_freq = keep_freq
        self.Q = Q
        self.b, self.a = iirpeak(self.keep_freq, self.Q, fs=self.data_freq)

    def __call__(self, sample_dict):
        x = sample_dict['x']
        return filtfilt(self.b, self.a, x)
    

class BandpassFilterStochastic:
    def __init__(self, data_freq=4, keep_freq_min=0.01, keep_freq_max=1, Q=0.707):
        """
        Randomly selects frequency band to keep
        :param keep_freq_min: min central frequency to keep
        :param keep_freq_max: max central frequency to keep
        :param Q: "quality factor" Q = width of filter
        """
        self.data_freq = data_freq
        self.keep_freq_min = keep_freq_min
        self.keep_freq_max = keep_freq_max
        self.Q = Q

    def __call__(self, sample_dict):
        x = sample_dict['x']
        keep_freq = np.random.uniform(self.keep_freq_min, self.keep_freq_max)
        b, a = iirpeak(keep_freq, self.Q, fs=self.data_freq)
        return filtfilt(b, a, x)


class TemporalCutoutDeterministic:
    """ Cutouts / Masks a section of the signal window """
    def __init__(self, cutout_size=100):
        self.cutout_size = cutout_size

    def __call__(self, sample_dict):
        x = sample_dict['x']
        # randomly sample cutout start
        start_min = 0
        start_max = len(x) - self.cutout_size
        cutout_start = np.random.choice(np.arange(start_min, start_max))
        x_trf = copy.deepcopy(x)
        x_trf[cutout_start:cutout_start+self.cutout_size] = 0
        return x_trf


class TemporalCutoutStochastic:
    """ Cutouts / Masks a section of the signal window """
    def __init__(self, cutout_size_min=1, cutout_size_max=120):
        self.cutout_size_min = cutout_size_min
        self.cutout_size_max = cutout_size_max

    def __call__(self, sample_dict):
        x = sample_dict['x']
        cutout_size = np.random.choice(np.arange(self.cutout_size_min, self.cutout_size_max+1))
        start_min = 0
        start_max = len(x) - cutout_size
        cutout_start = np.random.choice(np.arange(start_min, start_max+1))
        x_trf = copy.deepcopy(x)
        x_trf[cutout_start:cutout_start+cutout_size] = 0
        return x_trf


class PermuteDeterministic:
    """ Splits segments into chunks and permutes them """
    def __init__(self, n_splits=10):
        self.n_splits = n_splits

    def __call__(self, sample_dict):
        x = sample_dict['x']
        orig_steps = np.arange(x.shape[0])
        splits = np.array_split(orig_steps, self.n_splits)
        np.random.shuffle(splits)
        warp_idx = np.concatenate(splits)
        x_warped = x[warp_idx]
        return x_warped


class PermuteStochastic:
    """ Splits segments into chunks and permutes them """
    def __init__(self, n_splits_max=10):
        self.n_splits_max = n_splits_max

    def __call__(self, sample_dict):
        x = sample_dict['x']
        orig_steps = np.arange(x.shape[0])
        num_splits = np.random.randint(2, self.n_splits_max)
        splits = np.array_split(orig_steps, num_splits)
        np.random.shuffle(splits)
        warp_idx = np.concatenate(splits)
        x_warped = x[warp_idx]
        return x_warped


class Flip:
    """ Flips segments around horizontal axis """
    def __call__(self, sample_dict):
        x = sample_dict['x']
        flip = -1
        x_flip = flip * x + (2 * np.mean(x))
        return x_flip


class TimeShiftDeterministic:
    """ Shifts the window left or right by a number of samples """
    def __init__(self, shift_len=120):
        self.shift_len = shift_len

    def __call__(self, sample_dict):
        # compose x with left and right buffer
        left_buffer = sample_dict['x_left_buffer']
        right_buffer = sample_dict['x_right_buffer']
        # drop nans from left and right buffer segment['x_left_buffer']
        left_buffer = left_buffer[~np.isnan(left_buffer)]
        right_buffer = right_buffer[~np.isnan(right_buffer)]
        x = sample_dict['x']
        signal = np.concatenate([left_buffer, x, right_buffer])
        # sample shift to apply --- make sure not out-of-bounds!!
        left_shift_len = min(self.shift_len, len(left_buffer))
        right_shift_len = min(self.shift_len, len(right_buffer))
        shift = np.random.choice([-left_shift_len, right_shift_len])  # choose whether to shift left or right in time
        start_index = len(left_buffer) + shift
        x_trf = signal[start_index:start_index+len(x)]
        return x_trf


class TimeShiftStochastic:
    """ Shifts the window left or right by a number of samples """
    def __init__(self, shift_len_min=120, shift_len_max=240):
        self.shift_min = shift_len_min
        self.shift_max = shift_len_max
        self.shift_lens = np.arange(self.shift_min, self.shift_max, 1)

    def __call__(self, sample_dict):
        # compose x with left and right buffer
        left_buffer = sample_dict['x_left_buffer']
        right_buffer = sample_dict['x_right_buffer']
        # drop nans from left and right buffer segment['x_left_buffer']
        left_buffer = left_buffer[~np.isnan(left_buffer)]
        right_buffer = right_buffer[~np.isnan(right_buffer)]
        x = sample_dict['x']
        signal = np.concatenate([left_buffer, x, right_buffer])
        # sample shift len to apply
        shift_len = np.random.choice(self.shift_lens)
        # adjust so shift is in bounds & sample whether to apply it on left or right
        left_shift_len = min(shift_len, len(left_buffer))
        right_shift_len = min(shift_len, len(right_buffer))
        shift = np.random.choice([-left_shift_len, right_shift_len])  # choose whether to shift left or right in time
        start_index = len(left_buffer) + shift
        x_trf = signal[start_index:start_index+len(x)]
        return x_trf


class TimeWarpingDeterministic:
    """ 
    Warps the signal across time by creating a spline, warping time, 
    and then interpolating the signal back to the orignal time steps.
    See: https://arxiv.org/pdf/2007.15951.pdf
    """
    def __init__(self, sigma=0.25, knot=4):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, sample_dict):
        x = sample_dict['x']
        
        orig_steps = np.arange(x.shape[0])

        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot+2))
        warp_steps = (np.linspace(0, x.shape[0]-1, num=self.knot+2)).T

        # warps time to != 240
        time_warp = CubicSpline(warp_steps, warp_steps * random_warps)(orig_steps)  
        # scales the warping back to the window length
        scale = (x.shape[0]-1)/time_warp[-1]  
        # using the warped time steps and the original signal, linearly interpolate back onto the original time steps 
        x_time_warped = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[0]-1), x).T
        
        return x_time_warped
    

class TimeWarpingStochastic:
    """ 
    Warps the signal across time by creating a spline, warping time, 
    and then interpolating the signal back to the orignal time steps.
    See: https://arxiv.org/pdf/2007.15951.pdf
    """
    def __init__(self, sigma_min=0.01, sigma_max=0.25, knot_min=1, knot_max=4):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.knot_min = knot_min
        self.knot_max = knot_max

    def __call__(self, sample_dict):
        x = sample_dict['x']
        
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        knot = np.random.choice(np.arange(self.knot_min, self.knot_max+1))

        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
        warp_steps = (np.linspace(0, x.shape[0]-1, num=knot+2)).T

        # warps time to != 240
        time_warp = CubicSpline(warp_steps, warp_steps * random_warps)(orig_steps)  
        # scales the warping back to the window length
        scale = (x.shape[0]-1)/time_warp[-1]  
        # using the warped time steps and the original signal, linearly interpolate back onto the original time steps 
        x_time_warped = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[0]-1), x).T
        
        return x_time_warped


class ExtractComponent:
    def __init__(self, component, method="highpass"):
        self.component = component
        self.method = method

    def __call__(self, sample_dict):
        print("method", self.method)
        x = sample_dict['x']
        decomposed = nk.eda_phasic(x, sampling_rate=4, method=self.method)
        return decomposed[f"EDA_{self.component}"].to_numpy()


class ExtractPhasic(ExtractComponent):
    def __init__(self, method="highpass"):
        print("extract phasic")
        super().__init__("Phasic", method)


class ExtractTonic(ExtractComponent):
    def __init__(self, method="highpass"):
        print("extract tonic")
        super().__init__("Tonic", method)


class LooseSensorArtifactDeterministic:
    def __init__(self, width=4, smooth_width_min=2, smooth_width_max=80):
        self.width = width
        self.smooth_width_min = smooth_width_min
        self.smooth_width_max = smooth_width_max

    def __call__(self, sample_dict):
        x = sample_dict['x']
        
        # sample width of artifact
        artifact_width = self.width
        # sample artifact start
        artifact_start = np.random.choice(np.arange(0, len(x) - artifact_width + 1))
        # compute artifact end (inclusive)
        artifact_end = artifact_start + artifact_width - 1
        
        # don't smooth if artifact goes all the way to boundary
        smooth_left = (artifact_start != 0)
        smooth_right = (artifact_end != len(x) - 1)
        
        # sample smoothing edge widths
        smooth_max = min(self.smooth_width_max, int(artifact_width/2))
        smooth_width1 = np.random.choice(np.arange(self.smooth_width_min, smooth_max + 1)) if smooth_left else 0
        smooth_width2 = np.random.choice(np.arange(self.smooth_width_min, smooth_max + 1)) if smooth_right else 0
        
        # add drop to non-smoothed regions of artifact
        noisy_segment = copy.deepcopy(x)
        drop_start = artifact_start + smooth_width1
        drop_end = artifact_end - smooth_width2  # (inclusive)
        # get mean amplitude of signal in this range
        mean_amp = np.mean(noisy_segment[drop_start:drop_end + 1])  # +1 so inclusive
        # subtract from signal
        noisy_segment[drop_start:drop_end + 1] -= mean_amp
        # zero out negative entries
        noisy_segment[noisy_segment < 0] = 0
        
        # fill in parts to be smoothed
        # fit cubic spline
        # get pre-smooth, unsmoothed artifact, post-smooth
        train_x = np.concatenate([
            np.arange(artifact_start),  # don't include artifact start
            np.arange(drop_start, drop_end + 1),  # include drop end
            np.arange(artifact_end + 1, len(x)) # don't include artifact end
        ])
        train_y = np.concatenate([
            noisy_segment[:artifact_start],
            noisy_segment[drop_start:drop_end + 1],
            noisy_segment[artifact_end + 1:]
        ])
        spline = CubicSpline(train_x, train_y)
        # fill in smoothed parts
        if artifact_start != drop_start:
            noisy_segment[artifact_start:drop_start] = spline(np.arange(artifact_start, drop_start))
        if artifact_end != drop_end:
            noisy_segment[drop_end + 1:artifact_end + 1] = spline(np.arange(drop_end + 1, artifact_end + 1))  # include artifact end
        return noisy_segment


class LooseSensorArtifactStochastic:
    def __init__(self, width_min=4, width_max=120, smooth_width_min=2, smooth_width_max=20):
        self.width_min = width_min
        self.width_max = width_max
        self.smooth_width_min = smooth_width_min
        self.smooth_width_max = smooth_width_max

    def __call__(self, sample_dict):
        x = sample_dict['x']
        
        # sample width of artifact
        artifact_width = np.random.choice(np.arange(self.width_min, self.width_max + 1))  # +1 so its inclusive

        # sample artifact start
        artifact_start = np.random.choice(np.arange(0, len(x) - artifact_width + 1))
        # compute artifact end (inclusive)
        artifact_end = artifact_start + artifact_width - 1
        
        # don't smooth if artifact goes all the way to boundary
        smooth_left = (artifact_start != 0)
        smooth_right = (artifact_end != len(x) - 1)
        
        # sample smoothing edge widths
        smooth_available = int((artifact_width-2)/2)  # need to leave at least 2 dropped out samples
        smooth_max = min(self.smooth_width_max, smooth_available)
        smooth_min = min(smooth_max, self.smooth_width_min)
        smooth_width1 = np.random.choice(np.arange(smooth_min, smooth_max + 1)) if smooth_left else 0
        smooth_width2 = np.random.choice(np.arange(smooth_min, smooth_max + 1)) if smooth_right else 0
        
        # add drop to non-smoothed regions of artifact
        noisy_segment = copy.deepcopy(x)
        drop_start = artifact_start + smooth_width1
        drop_end = artifact_end - smooth_width2  # (inclusive)
        # get mean amplitude of signal in this range
        mean_amp = np.mean(noisy_segment[drop_start:drop_end + 1])  # +1 so inclusive
        # subtract from signal
        noisy_segment[drop_start:drop_end + 1] -= mean_amp
        # zero out negative entries
        noisy_segment[noisy_segment < 0] = 0
        
        # fill in parts to be smoothed
        # fit cubic spline
        # get pre-smooth, unsmoothed artifact, post-smooth
        train_x = np.concatenate([
            np.arange(artifact_start),  # don't include artifact start
            np.arange(drop_start, drop_end + 1),  # include drop end
            np.arange(artifact_end + 1, len(x)) # don't include artifact end
        ])
        train_y = np.concatenate([
            noisy_segment[:artifact_start],
            noisy_segment[drop_start:drop_end + 1],
            noisy_segment[artifact_end + 1:]
        ])
        spline = CubicSpline(train_x, train_y)
        # fill in smoothed parts
        if artifact_start != drop_start:
            noisy_segment[artifact_start:drop_start] = spline(np.arange(artifact_start, drop_start))
        if artifact_end != drop_end:
            noisy_segment[drop_end + 1:artifact_end + 1] = spline(np.arange(drop_end + 1, artifact_end + 1))  # include artifact end
        return noisy_segment


class JumpArtifactDeterministic:
    def __init__(self, max_n_jumps=2, shift_factor=0.1, smooth_width_min=2, smooth_width_max=12):
        self.max_n_jumps = max_n_jumps
        self.shift_factor = shift_factor
        self.smooth_width_min = smooth_width_min
        self.smooth_width_max = smooth_width_max

    def __call__(self, sample_dict):
        x = sample_dict['x']
        noisy_segment = copy.deepcopy(x)
        
        # time flip so we can apply the logic below in either direction
        time_flip = np.random.choice([-1, 1]) 
        if time_flip == -1:
            noisy_segment = np.flip(noisy_segment)
        
        # sample n artifacts
        n_jumps = np.random.choice(np.arange(1, self.max_n_jumps + 1)) # make inclusive
        
        # sample artifact starts and shift factors
        min_start = 1 # don't start at 0 because this would shift whole segment instead of creating jump
        # needs to start early enough that there is enough room to smooth jump (with smallest smoothing window)
        max_start = len(x) - self.smooth_width_min - 2 
        artifact_starts = np.sort(np.random.choice(np.arange(min_start, max_start + 1), size=n_jumps, replace=False))
        artifact_shift_factors = self.shift_factor * np.random.choice([-1, 1], size=n_jumps)
        
        # loop through & apply shifts
        for idx, a_start in enumerate(artifact_starts):
            # sample smoothing window (how many samples to smooth)
            # smooth window needs to fit in between a_start and end of x with at least a one sample gap
            _smooth_max = min(self.smooth_width_max, len(x) - a_start - 2)
            a_smooth_win = np.random.choice(np.arange(self.smooth_width_min, _smooth_max + 1)) # make inclusive
            x_post_smooth = noisy_segment[a_start + a_smooth_win:]
            # add jump to x_post_smooth, scale it by width of smooth window (want to control jump/sec)
            x_post_smooth += artifact_shift_factors[idx] * (a_smooth_win / 4)  # get smooth win in secs
            
            # fill in parts to be smoothed
            # fit cubic spline
            # get pre-smooth, unsmoothed artifact, post-smooth
            train_x = np.concatenate([
                np.arange(a_start),  # everywhere but where smoothing occurs
                np.arange(a_start + a_smooth_win, len(x))
            ])
            train_y = np.concatenate([
                noisy_segment[:a_start],
                noisy_segment[a_start + a_smooth_win:],
            ])
            spline = CubicSpline(train_x, train_y)
            # fill in smoothed parts
            noisy_segment[a_start:a_start + a_smooth_win] = spline(np.arange(a_start, a_start + a_smooth_win))
            
            # zero out negative entries
            noisy_segment[noisy_segment < 0] = 0
            
        # Flip the segment back to original time order if it was reversed
        if time_flip == -1:
            noisy_segment = np.flip(noisy_segment)
        
        return noisy_segment
    

class JumpArtifactStochastic:
    def __init__(self, max_n_jumps=2, shift_factor_min=0.01, shift_factor_max=0.2, smooth_width_min=2, smooth_width_max=12):
        self.max_n_jumps = max_n_jumps
        self.shift_factor_min = shift_factor_min
        self.shift_factor_max = shift_factor_max
        self.smooth_width_min = smooth_width_min
        self.smooth_width_max = smooth_width_max

    def __call__(self, sample_dict):
        x = sample_dict['x']
        noisy_segment = copy.deepcopy(x)
        
        # time flip so we can apply the logic below in either direction
        time_flip = np.random.choice([-1, 1]) 
        if time_flip == -1:
            noisy_segment = np.flip(noisy_segment)
        
        # sample n artifacts
        n_jumps = np.random.choice(np.arange(1, self.max_n_jumps + 1)) # make inclusive
        
        # sample artifact starts and shift factors
        min_start = 1 # don't start at 0 because this would shift whole segment instead of creating jump
        # needs to start early enough that there is enough room to smooth jump (with smallest smoothing window)
        max_start = len(x) - self.smooth_width_min - 2 
        artifact_starts = np.sort(np.random.choice(np.arange(min_start, max_start + 1), size=n_jumps, replace=False))
        artifact_shift_factors = np.random.uniform(low=self.shift_factor_min, high=self.shift_factor_max, size=n_jumps) * np.random.choice([-1, 1], size=n_jumps)
        
        # loop through & apply shifts
        for idx, a_start in enumerate(artifact_starts):
            # sample smoothing window (how many samples to smooth)
            # smooth window needs to fit in between a_start and end of x with at least a one sample gap
            _smooth_max = min(self.smooth_width_max, len(x) - a_start - 2)
            a_smooth_win = np.random.choice(np.arange(self.smooth_width_min, _smooth_max + 1)) # make inclusive
            x_pre_artifact = noisy_segment[:a_start]
            x_smooth_win = noisy_segment[a_start:a_start + a_smooth_win]
            x_post_smooth = noisy_segment[a_start + a_smooth_win:]
            # add jump to x_post_smooth, scale it by width of smooth window (want to control jump/sec)
            x_post_smooth += artifact_shift_factors[idx] * (a_smooth_win / 4)  # get smooth win in secs
            
            # fill in parts to be smoothed
            # fit cubic spline
            # get pre-smooth, unsmoothed artifact, post-smooth
            train_x = np.concatenate([
                np.arange(a_start),  # everywhere but where smoothing occurs
                np.arange(a_start + a_smooth_win, len(x))
            ])
            train_y = np.concatenate([
                noisy_segment[:a_start],
                noisy_segment[a_start + a_smooth_win:],
            ])
            spline = CubicSpline(train_x, train_y)
            # fill in smoothed parts
            noisy_segment[a_start:a_start + a_smooth_win] = spline(np.arange(a_start, a_start + a_smooth_win))
            
            # zero out negative entries
            noisy_segment[noisy_segment < 0] = 0
            
        # Flip the segment back to original time order if it was reversed
        if time_flip == -1:
            noisy_segment = np.flip(noisy_segment)

        return noisy_segment


class ConstantAmplitudeScalingDeterministic:
    """ Scale EDA by constant factor across the window """
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, sample_dict):
        x = sample_dict['x']
        return x * self.scale
    
    
class ConstantAmplitudeScalingStochastic:
    """ Scale EDA by constant factor across the window """
    def __init__(self, scale_min=0.5, scale_max=1.5):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, sample_dict):
        x = sample_dict['x']
        scale = np.random.uniform(self.scale_min, self.scale_max)
        return x * scale


class AmplitudeWarpingDeterministic:
    """ 
    Scale EDA by a smoothly varying factor across the window 
    Note: if knot = 0 then scale factor changes linearly
    See: https://arxiv.org/pdf/2007.15951.pdf
    """
    def __init__(self, sigma=0.2, knot=4):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, sample_dict):
        x = sample_dict['x']
        
        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot+2))
        warp_steps = (np.linspace(0, x.shape[0]-1, num=self.knot+2)).T

        warper = CubicSpline(warp_steps, random_warps)(orig_steps)        

        return x * warper
    

class AmplitudeWarpingStochastic:
    """ 
    Scale EDA by a smoothly varying factor across the window 
    Note: if knot = 0 then scale factor changes linearly
    See: https://arxiv.org/pdf/2007.15951.pdf
    """
    def __init__(self, sigma_min=0.01, sigma_max=0.25, knot_min=0, knot_max=4):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.knot_min = knot_min
        self.knot_max = knot_max

    def __call__(self, sample_dict):
        x = sample_dict['x']
        
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        knot = np.random.choice(np.arange(self.knot_min, self.knot_max+1))

        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
        warp_steps = (np.linspace(0, x.shape[0]-1, num=knot+2)).T

        warper = CubicSpline(warp_steps, random_warps)(orig_steps)        

        return x * warper


class TonicConstantAmplitudeScalingDeterministic:
    """
    Mimics the effect of temperature / humidity on the EDA signal with a CONSTANT scale factor on tonic
    See: Qasim, Masood S., Dindar S. Bari, and Ørjan G. Martinsen. 
    “Influence of ambient temperature on tonic and phasic electrodermal activity components.” 
    Physiological Measurement 43.6 (2022): 065001.
    And: Bari, D. S., Aldosky, H. Y. Y., Tronstad, C., Kalvøy, H., & Martinsen, Ø. G. (2018). 
    "Influence of relative humidity on electrodermal levels and responses." 
    Skin pharmacology and physiology, 31(6), 298-307.
    """
    def __init__(self, tonic_scale_factor, data_freq=4):
        self.tonic_scale_factor = tonic_scale_factor
        # design the filters to extract tonic and phasic
        self.data_freq = data_freq
        self.b_tonic, self.a_tonic = scipy.signal.butter(4, [0.05], btype="lowpass", output="ba", fs=data_freq) 
        self.b_phasic, self.a_phasic = scipy.signal.butter(4, [0.05], btype="highpass", output="ba", fs=data_freq) 

    def __call__(self, sample_dict):
        
        x = sample_dict['x']
        tonic = scipy.signal.filtfilt(self.b_tonic, self.a_tonic, x)
        phasic = scipy.signal.filtfilt(self.b_phasic, self.a_phasic, x)
        tonic_scaled = tonic * self.tonic_scale_factor
        
        return tonic_scaled + phasic
    
    
class TonicConstantAmplitudeScalingStochastic:
    """
    Mimics the effect of temperature / humidity on the EDA signal with a CONSTANT scale factor on tonic
    See: Qasim, Masood S., Dindar S. Bari, and Ørjan G. Martinsen. 
    “Influence of ambient temperature on tonic and phasic electrodermal activity components.” 
    Physiological Measurement 43.6 (2022): 065001.
    And: Bari, D. S., Aldosky, H. Y. Y., Tronstad, C., Kalvøy, H., & Martinsen, Ø. G. (2018). 
    "Influence of relative humidity on electrodermal levels and responses." 
    Skin pharmacology and physiology, 31(6), 298-307.
    """
    def __init__(self, tonic_scale_factor_min=0.5, tonic_scale_factor_max=1.5, data_freq=4):
        self.tonic_scale_factor_min = tonic_scale_factor_min
        self.tonic_scale_factor_max = tonic_scale_factor_max
        # design the filters to extract tonic and phasic. Note: it's OK for these to be in constructor as they're constant
        self.data_freq = data_freq
        self.b_tonic, self.a_tonic = scipy.signal.butter(4, [0.05], btype="lowpass", output="ba", fs=data_freq) 
        self.b_phasic, self.a_phasic = scipy.signal.butter(4, [0.05], btype="highpass", output="ba", fs=data_freq) 

    def __call__(self, sample_dict):
        x = sample_dict['x']
        
        tonic_scale_factor = np.random.uniform(self.tonic_scale_factor_min, self.tonic_scale_factor_max)
        
        tonic = scipy.signal.filtfilt(self.b_tonic, self.a_tonic, x)
        phasic = scipy.signal.filtfilt(self.b_phasic, self.a_phasic, x)
        tonic_scaled = tonic * tonic_scale_factor
        
        return tonic_scaled + phasic


class TonicAmplitudeWarpingDeterministic:
    """
    Mimics the effect of temperature / humidity on the EDA signal with a SMOOTH time-varying scale factor on tonic
    See: Qasim, Masood S., Dindar S. Bari, and Ørjan G. Martinsen. 
    “Influence of ambient temperature on tonic and phasic electrodermal activity components.” 
    Physiological Measurement 43.6 (2022): 065001.
    And: Bari, D. S., Aldosky, H. Y. Y., Tronstad, C., Kalvøy, H., & Martinsen, Ø. G. (2018). 
    "Influence of relative humidity on electrodermal levels and responses." 
    Skin pharmacology and physiology, 31(6), 298-307.
    """
    def __init__(self, sigma=0.1, knot=4, data_freq=4):
        self.sigma = sigma
        self.knot = knot
        # design the filters to extract tonic and phasic
        self.data_freq = data_freq
        self.b_tonic, self.a_tonic = scipy.signal.butter(4, [0.05], btype="lowpass", output="ba", fs=data_freq) 
        self.b_phasic, self.a_phasic = scipy.signal.butter(4, [0.05], btype="highpass", output="ba", fs=data_freq) 

    def __call__(self, sample_dict):
        
        x = sample_dict['x']
        
        tonic = scipy.signal.filtfilt(self.b_tonic, self.a_tonic, x)
        phasic = scipy.signal.filtfilt(self.b_phasic, self.a_phasic, x)
        
        orig_steps = np.arange(tonic.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(self.knot+2))
        warp_steps = (np.linspace(0, x.shape[0]-1, num=self.knot+2)).T
        warper = CubicSpline(warp_steps, random_warps)(orig_steps)
        tonic_warped = tonic * warper
        
        return tonic_warped + phasic
    
    
class TonicAmplitudeWarpingStochastic:
    """
    Mimics the effect of temperature / humidity on the EDA signal with a SMOOTH time-varying scale factor on tonic
    See: Qasim, Masood S., Dindar S. Bari, and Ørjan G. Martinsen. 
    “Influence of ambient temperature on tonic and phasic electrodermal activity components.” 
    Physiological Measurement 43.6 (2022): 065001.
    And: Bari, D. S., Aldosky, H. Y. Y., Tronstad, C., Kalvøy, H., & Martinsen, Ø. G. (2018). 
    "Influence of relative humidity on electrodermal levels and responses." 
    Skin pharmacology and physiology, 31(6), 298-307.
    """
    def __init__(self, sigma_min=0.01, sigma_max=0.25, knot_min=0, knot_max=4, data_freq=4):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.knot_min = knot_min
        self.knot_max = knot_max
        # design the filters to extract tonic and phasic. Note: it's OK for these to be in constructor as they're constant
        self.data_freq = data_freq
        self.b_tonic, self.a_tonic = scipy.signal.butter(4, [0.05], btype="lowpass", output="ba", fs=data_freq) 
        self.b_phasic, self.a_phasic = scipy.signal.butter(4, [0.05], btype="highpass", output="ba", fs=data_freq) 

    def __call__(self, sample_dict):
        x = sample_dict['x']
        
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        knot = np.random.choice(np.arange(self.knot_min, self.knot_max+1))

        tonic = scipy.signal.filtfilt(self.b_tonic, self.a_tonic, x)
        phasic = scipy.signal.filtfilt(self.b_phasic, self.a_phasic, x)
        
        orig_steps = np.arange(tonic.shape[0])
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
        warp_steps = (np.linspace(0, x.shape[0]-1, num=knot+2)).T
        warper = CubicSpline(warp_steps, random_warps)(orig_steps)
        tonic_warped = tonic * warper
        
        return tonic_warped + phasic


"""
Deprecated below this line 
"""

class FlipWrist:
    def __call__(self, sample_dict):
        x_opp_wrist = sample_dict['x_opp_wrist']
        return x_opp_wrist


class ScalePhasic:
    def __init__(self, scale_factor_max=2, scale_step=0.1):
        self.scale_factors = np.arange(1 / scale_factor_max, scale_factor_max + scale_step, scale_step)

    def __call__(self, sample_dict):
        x = sample_dict['x']
        # randomly sample scaling factor
        scale = np.random.choice(self.scale_factors)
        decomp = nk.eda_phasic(x, sampling_rate=4, method='highpass')
        phasic = decomp["EDA_Phasic"].to_numpy()
        phasic_scaled = phasic * scale
        return decomp["EDA_Tonic"].to_numpy() + phasic_scaled


class DCShift:
    def __init__(self, shift_factor_max=2, shift_step=0.1):
        self.shift_factors = np.arange(-shift_factor_max, shift_factor_max, shift_step)

    def __call__(self, sample_dict):
        x = sample_dict['x']
        # randomly sample shift
        shift_factor = np.random.choice(self.shift_factors)
        mean_power_diff = np.mean(np.abs(x - np.mean(x)))
        shift = shift_factor * mean_power_diff
        return x + shift
