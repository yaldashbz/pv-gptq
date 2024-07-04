import torch
import torch.nn as nn

from typing import List, Optional, Union, Tuple
from src.kmeans import find_nearest_cluster, fit_faiss_kmeans, fit_kmeans, fit_kmeans_1d
from src.beam_search_xtx import beam_search_optimal_codes as beam_search_minimize_activation_mse
from src.beam_search_l2 import beam_search_optimal_codes as beam_search_minimize_weight_mse
from src.aq_ops import IntCodes

# Copied from https://github.com/IST-DASLab/marlin/pull/1
@torch.no_grad()
def unpack_4bit_to_32bit_signed(qweight, qzeros):
    # Unpack 4-bit values and interpret them as signed integers
    unpacked_weights = torch.zeros(
        (qweight.shape[0] * 8, qweight.shape[1]),
        dtype=torch.int8,
        device=qweight.device,
        requires_grad=False,
    )

    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * 8),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for row in range(unpacked_weights.shape[0]):
        i = row % 8
        unpacked_weights[row, :] = (qweight[row // 8, :] >> (4 * i)) & 0xF

    for col in range(unpacked_zeros.shape[1]):
        i = col % 8
        unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

    return unpacked_weights, unpacked_zeros + 1

def unpack_qzeros(qzeros):
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * 8),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % 8
        unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

    return unpacked_zeros + 1


# Copied from https://github.com/IST-DASLab/marlin/pull/1
@torch.no_grad()
def dequantize_weight(layer):
    qweight, qzeros, scales = layer.qweight, layer.qzeros, layer.scales
    unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)
    group_size = unpacked_qweight.shape[0] // scales.shape[0]
    scales = scales.repeat_interleave(group_size, dim=0)
    unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
    unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales

    return unpacked_qweight.T, unpacked_qzeros



class GPTQQuantizedWeight(nn.Module):
    def __init__(
            self,
            refrence_layer, # quantized refrence layer (int32)
            scale_nbits: int = 0,
            straight_through_gradient: Optional[bool] = None
) -> None:
        super().__init__()
        self.scales = nn.Parameter(refrence_layer.scales, requires_grad=True)
        self.qzeros = nn.Parameter(refrence_layer.qzeros, requires_grad=False)
        self.qweight = nn.Parameter(refrence_layer.qweight, requires_grad=False)
        self.qweight_storage: Optional[IntCodes] = None  # storage for FSDP compatibility
        self.qzeros_storage: Optional[IntCodes] = None  # storage for FSDP compatibility

        dequantized_weight, _ = dequantize_weight(refrence_layer)
        self.out_features, self.in_features = dequantized_weight.shape
        self.scale_nbits = scale_nbits
        self.straight_through_gradient = straight_through_gradient
        self.scales_are_lossless = scale_nbits == 0
    
    @property
    def shape(self):
        return self.out_features, self.in_features
    
    def forward(self):
        dequantized_weight = self.dequantize_weight(self.qweight, self.qzeros, self.scales)
        dequantized_weight.requires_grad_(True)
        return dequantized_weight
    
    def dequantize_weight(self, qweight, qzeros, scales):
        unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)
        group_size = unpacked_qweight.shape[0] // scales.shape[0]
        scales = scales.repeat_interleave(group_size, dim=0)
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
        unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales
        return unpacked_qweight.T

    def estimate_nbits_per_parameter(self) -> float:
        """Calculate the effective number of bits per original matrix parameters"""
        return 4
    
    def wrap_params_for_fsdp_(self, **kwargs):
        """Make this module compatible with FullyShardedDataParallel; modifies state dict in-place"""
        assert self.qweight is not None and self.qweight_storage is None
        self.qweight_storage, self.qweight = IntCodes(self.qweight, **kwargs), None

        assert self.qzeros is not None and self.qzeros_storage is None
        self.qzeros_storage, self.qzeros = IntCodes(self.qzeros, **kwargs), None
    
    def unwrap_params_(self):
        """Undo the effect of wrap_params_for_fsdp_; modifies state dict in-place"""
        assert self.qweight is None and self.qweight_storage is not None
        self.qweight, self.qweight_storage = nn.Parameter(self.qweight_storage(), requires_grad=False), None

        assert self.qzeros is None and self.qzeros_storage is not None
        self.qzeros, self.qzeros_storage = nn.Parameter(self.qzeros_storage(), requires_grad=False), None
    
    def get_qweight(self) -> torch.IntTensor:
        """Get a non view to qweight, regardless of how codes are stored"""
        assert (self.qweight is None) != (self.qweight_storage is None), "must have either .codes or storage, but not both"
        qweight = self.qweight if self.qweight is not None else self.qweight_storage()
        if torch.iinfo(qweight.dtype).bits < 32:
            qweight = qweight.to(torch.int32)  # cast to int32 to allow indexing if codes are int16 or uint8
        return qweight
    
    def get_qzeros(self) -> torch.IntTensor:
        """Get a non view to qzeros, regardless of how codes are stored"""
        assert (self.qzeros is None) != (self.qzeros_storage is None), "must have either .codes or storage, but not both"
        qzeros = self.qzeros if self.qzeros is not None else self.qzeros_storage()
        if torch.iinfo(qzeros.dtype).bits < 32:
            qzeros = qzeros.to(torch.int32)  # cast to int32 to allow indexing if codes are int16 or uint8
        return qzeros

    def get_scales(self) -> torch.Tensor:
        """Get per-channel or per-group quantization scales or reconstruct those scales based on scales_nbits"""
        if self.scale_nbits == 0 or self.scales_are_lossless:
            return self.scales  # scales are not quantized or the quantization is lossless
        elif self.straight_through_gradient:
            with torch.no_grad():
                self.scales_clusters, _, dequantized_scales = fit_kmeans_1d(
                    self.scales.flatten(1, -1), k=2**self.scale_nbits, initial_clusters=self.scales_clusters
                )
                dequantized_scales = dequantized_scales.reshape_as(self.scales)
            if torch.is_grad_enabled() and self.scales.requires_grad:
                dequantized_scales = dequantized_scales + (self.scales - self.scales.detach())
            return dequantized_scales
        else:  # train scale codebook only
            return self.scales_clusters.gather(1, self.scales_indices)[:, :, None, None]

    def get_discretes(self) -> torch.Tensor:
        qweight, qzeros = self.get_qweight(), self.get_qzeros()
        discretes = torch.cat((qweight.flatten(), qzeros.flatten()))
        return discretes

    def get_codes(self):
        # TODO move to wrapper
        return self.get_discretes()
