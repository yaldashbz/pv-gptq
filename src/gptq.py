import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

from abc import ABC, abstractmethod

from typing import Optional
from src.aq_ops import IntCodes
from src.gptq_ops import (
    get_discrete_group_indices, unpack_4bit_to_32bit_signed, 
    pack_32bit_to_4bit, undo_repeat_interleave
)


class GPTQQuantizedWeight(ABC, nn.Module):
    @staticmethod
    def create_quantized_weight(quant_class, reference_layer):
        return quant_class(reference_layer)

    @staticmethod
    def get_quant_class(model):
        for _, module in model.named_modules():
            if 'QuantLinear' in str(module.__class__):
                return QuantLinearQuantizedWeight, 'QuantLinear'
            if 'LinearNF4' in str(module.__class__):
                return LinearNF4QuantizedWeight, 'LinearNF4'
        raise ValueError("Not quantized! or not supported method.")
    
    def estimate_nbits_per_parameter(self) -> float:
        # TODO
        """Calculate the effective number of bits per original matrix parameters"""
        return 4
    
    @abstractmethod
    def wrap_params_for_fsdp_(self, **kwargs):
        """Make this  compatible with FullyShardedDataParallel; modifies state dict in-place"""
        raise NotImplementedError
    
    @abstractmethod
    def unwrap_params_(self):
        """Undo the effect of wrap_params_for_fsdp_; modifies state dict in-place"""
        raise NotImplementedError
    
    @abstractmethod
    def update_discretes(self, reference_weight, max_update_fraction, lr):
        raise NotImplementedError



class LinearNF4QuantizedWeight(GPTQQuantizedWeight):
    def __init__(
            self,
            refrence_layer, # quantized refrence layer (NF4)
    ) -> None:
        super().__init__()
        # self.quant_state = refrence_layer.weight.quant_state
        self.absmax = nn.Parameter(refrence_layer.weight.quant_state.absmax, requires_grad=True)
        self.nf4weight = nn.Parameter(refrence_layer.weight.data, requires_grad=False)
        self.nf4weight_storage: Optional[IntCodes] = None  # storage for FSDP compatibility

        self.out_features, self.in_features = refrence_layer.weight.quant_state.shape
        self.blocksize = refrence_layer.weight.quant_state.blocksize
        self.dtype = refrence_layer.weight.quant_state.dtype
    
    @property
    def shape(self):
        return self.out_features, self.in_features
    
    def forward(self):
        # assert self.quant_state.absmax.requires_grad
        nf4weight = self.get_nf4weight()
        dequantized_weight = torch.empty(
            (self.out_features, self.in_features), dtype=self.dtype, device=nf4weight.device, requires_grad=True)
        bnb.functional.dequantize_nf4(
            A=nf4weight, 
            absmax=self.absmax,
            out=dequantized_weight,
            blocksize=self.blocksize
        )
        return dequantized_weight
    
    def wrap_params_for_fsdp_(self, **kwargs):
        """Make this  compatible with FullyShardedDataParallel; modifies state dict in-place"""
        assert self.nf4weight is not None and self.nf4weight_storage is None
        self.nf4weight_storage, self.nf4weight = IntCodes(self.nf4weight, **kwargs), None
    
    def unwrap_params_(self):
        """Undo the effect of wrap_params_for_fsdp_; modifies state dict in-place"""
        assert self.nf4weight is None and self.nf4weight_storage is not None
        self.nf4weight, self.nf4weight_storage = nn.Parameter(self.nf4weight_storage(), requires_grad=False), None
    
    def get_nf4weight(self) -> torch.IntTensor:
        """Get a non view to qweight, regardless of how codes are stored"""
        assert (self.nf4weight is None) != (self.nf4weight_storage is None), "must have either .codes or storage, but not both"
        nf4weight = self.nf4weight if self.nf4weight is not None else self.nf4weight_storage()
        if torch.iinfo(nf4weight.dtype).bits < 32:
            nf4weight = nf4weight.to(torch.int32)  # cast to int32 to allow indexing if codes are int16 or uint8
        return nf4weight
    
    def set_nf4weight(self, new_nf4weight: torch.Tensor, **kwargs):
        """Update codes[selection] to new_codes, regardless of their dtype and whether they are wrapped as storage"""
        assert (self.nf4weight is None) != (self.nf4weight_storage is None), "must have either .codes or storage, but not both"
        nf4weight = self.nf4weight if self.nf4weight is not None else self.nf4weight_storage()
        nf4weight.copy_(new_nf4weight, **kwargs)

    def update_discretes(self, reference_weight, max_update_fraction, lr):
        pass


class QuantLinearQuantizedWeight(GPTQQuantizedWeight):
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
        dequantized_weight = self.dequantize_weight(
            self.get_qweight(), self.get_qzeros(), self.get_scales())
        if self.out_features is None:
            self.out_features, self.in_features = dequantized_weight.shape
        return dequantized_weight
    
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
        
        qweight, qzeros = self.update_qweight_qzeros(
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

    def dequantize_weight(self, qweight, qzeros, scales):
        unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)
        group_size = unpacked_qweight.shape[0] // scales.shape[0]
        scales = scales.repeat_interleave(group_size, dim=0)
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
        unpacked_qweight = (unpacked_qweight - unpacked_qzeros) * scales
        return unpacked_qweight.T

    def update_qweight_qzeros(self, prev_qweight, prev_qzeros, new_scales, reference_weight, max_update_fraction, lr):
        unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(prev_qweight, prev_qzeros)
        group_size = unpacked_qweight.shape[0] // new_scales.shape[0]
        scales = new_scales.repeat_interleave(group_size, dim=0)
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
        qweight_grad = reference_weight.grad.T * scales
        qzeros_grad = -reference_weight.grad.T * scales

        """Update only topk
        qweight = unpacked_qweight - lr * qweight_grad
        qzeros = unpacked_qzeros - lr * qzeros_grad
        """
        in_group_size, out_group_size = group_size, 1   # for GPTQ
        flat_indices_to_update = get_discrete_group_indices(
            reference_weight, out_group_size, in_group_size, max_update_fraction)
        
        def _update_discrete_param(group_indices, targeted_tensor, targeted_tensor_grad):
            num_groups = reference_weight.shape[1] // group_size

            # Calculate row and group positions in w
            row_indices = group_indices // num_groups
            group_positions = group_indices % num_groups

            # Calculate the start and end positions in q
            start_cols = group_positions * group_size
            end_cols = start_cols + group_size

            for row, start_col, end_col in zip(row_indices, start_cols, end_cols):
                targeted_tensor[start_col:end_col, row] = targeted_tensor[start_col:end_col, row] - lr * targeted_tensor_grad[start_col:end_col, row]
            
            return targeted_tensor
        

        qweight = _update_discrete_param(flat_indices_to_update, unpacked_qweight.clone().float(), qweight_grad).round().int()
        qzeros = _update_discrete_param(flat_indices_to_update, unpacked_qzeros.clone().float(), qzeros_grad).round().int()

        qweight, qzeros = pack_32bit_to_4bit(qweight, qzeros)
        qzeros = undo_repeat_interleave(qzeros, group_size, dim=0)

        assert qweight.shape == prev_qweight.shape
        assert qzeros.shape == prev_qzeros.shape
        return qweight, qzeros
