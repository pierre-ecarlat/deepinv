import numpy as np
import torch

from deepinv.physics.forward import LinearPhysics

# Not sure how you want to import it as it shouldn't be a requirement
import cupy as cp
from ultraspy.beamformers.das import DelayAndSum
from ultraspy.scan import GridScan
from ultraspy.probes.factory import get_probe
import ultraspy as us


class UltrasoundBeamforming(LinearPhysics):
    r"""
    Performs a Delay And Sum beamforming operator for ultrasound imaging.

    :param tuple[int] img_size: size of the input raw data, of shape
        (nb_cycles, nb_probe_elements, nb_time_samples)
    :param dict acquisition_info: the information about the acquisition setup,
        including a minima the sampling_freq and delays. Optionally the t0,
        prf, signal_duration, sound_speed.
    :param str probe_name: the name of the probe
    :param np.array x_axis: the lateral axis of the medium to observe
    :param np.array y_axis: the elevational axis of the medium to observe
    :param np.array z_axis: the axial axis of the medium to observe
    :param str device: Device to store the forward matrix
    """

    def __init__(
        self,
        img_size,
        acquisition_info,
        probe_name,
        x_axis,
        y_axis,
        z_axis,
        device="cpu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.imsize = img_size

        on_gpu = (device == 'gpu')
        self.probe = get_probe(probe_name)
        self.scan = GridScan(x_axis, y_axis, z_axis)

        self.beamformer = DelayAndSum(on_gpu=on_gpu)
        self.beamformer.automatic_setup(acquisition_info, probe_name)

    def A(self, x):
        r"""
        The forward method to perform beamforming.

        :param torch.Tensor x: The raw RFs data, I guess in pytorch?
        """
        if self.beamformer.on_gpu:
            raw_data = cp.asarray(x, np.float32)
        else:
            raw_data = x.detach().numpy()

        # If working on IQs, should convert here, a bit different on CPU / GPU

        beamformed = self.beamformer.beamform(raw_data, self.scan)
        envelope = self.beamformer.compute_envelope(beamformed, self.scan)

        # Transfer to torch
        if self.beamformer.on_gpu:
            envelope = torch.as_tensor(envelope, device='cuda')
        else:
            envelope = torch.as_tensor(envelope, device='cpu')

        return envelope
