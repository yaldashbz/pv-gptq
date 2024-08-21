import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from src.aq_ops import IntCodes
from src.gptq_ops import *


methods = {
    'quantlin': QuantLinear(),
    'nf4': None
}


class GPTQQuantizedWeight(nn.Module):
    def __init__(
            self,
            refrence_layer, # quantized refrence layer (int32)
            scale_nbits: int = 0,
            straight_through_gradient: Optional[bool] = None,
            quant_method: str = 'quantlin'
) -> None:
        super().__init__()
        self.scales = nn.Parameter(refrence_layer.scales, requires_grad=True)
        self.qzeros = nn.Parameter(refrence_layer.qzeros, requires_grad=False)
        self.qweight = nn.Parameter(refrence_layer.qweight, requires_grad=False)
        self.qweight_storage: Optional[IntCodes] = None  # storage for FSDP compatibility
        self.qzeros_storage: Optional[IntCodes] = None  # storage for FSDP compatibility
        
        self._quant_method = methods[quant_method]
        self.out_features, self.in_features = None, None
        self.scale_nbits = scale_nbits
        self.straight_through_gradient = straight_through_gradient
        self.scales_are_lossless = scale_nbits == 0
    
    @property
    def shape(self):
        assert self.out_features and self.in_features
        return self.out_features, self.in_features

    def freeze_scales(self):
        self.scales.requires_grad = False

    def forward(self):
        dequantized_weight = self._quant_method.dequantize_weight(
            self.get_qweight(), self.get_qzeros(), self.get_scales())
        if self.out_features is None:
            self.out_features, self.in_features = dequantized_weight.shape
        return dequantized_weight

    def estimate_nbits_per_parameter(self) -> float:
        # TODO
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
        return self.scales
        # if self.scale_nbits == 0 or self.scales_are_lossless:
        #     return self.scales  # scales are not quantized or the quantization is lossless
        # elif self.straight_through_gradient:
        #     with torch.no_grad():
        #         self.scales_clusters, _, dequantized_scales = fit_kmeans_1d(
        #             self.scales.flatten(1, -1), k=2**self.scale_nbits, initial_clusters=self.scales_clusters
        #         )
        #         dequantized_scales = dequantized_scales.reshape_as(self.scales)
        #     if torch.is_grad_enabled() and self.scales.requires_grad:
        #         dequantized_scales = dequantized_scales + (self.scales - self.scales.detach())
        #     return dequantized_scales
        # else:  # train scale codebook only
        #     return self.scales_clusters.gather(1, self.scales_indices)[:, :, None, None]
        
    def update_discretes(self, reference_weight, max_update_fraction, lr):
        prev_qweight = self.get_qweight().clone()
        prev_qzeros = self.get_qzeros().clone()
        scales = self.get_scales().clone()
        
        qweight, qzeros = self._quant_method.update_discretes(
            prev_qweight, prev_qzeros, scales, reference_weight, max_update_fraction, lr)

        self.set_qweight(qweight)
        self.set_qzeros(qzeros)
        return qweight, qzeros

    def set_qweight(self, new_qweight: torch.Tensor, **kwargs):
        """Update codes[selection] to new_codes, regardless of their dtype and whether they are wrapped as storage"""
        assert (self.qweight is None) != (self.qweight_storage is None), "must have either .codes or storage, but not both"
        qweight = self.qweight if self.qweight is not None else self.qweight_storage()
        qweight.copy_(new_qweight, **kwargs)

    def set_qzeros(self, new_qzeros: torch.Tensor, **kwargs):
        """Update codes[selection] to new_codes, regardless of their dtype and whether they are wrapped as storage"""
        assert (self.qzeros is None) != (self.qzeros_storage is None), "must have either .codes or storage, but not both"
        qzeros = self.qzeros if self.qzeros is not None else self.qzeros_storage()
        qzeros.copy_(new_qzeros, **kwargs)

