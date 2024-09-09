import torch
import math


def get_discrete_group_indices(
        reference_weight: torch.Tensor, 
        out_group_size: int,
        in_group_size: int,
        max_update_fraction: float = .01
    ):
        num_output_groups = reference_weight.shape[0] // out_group_size
        num_input_groups = reference_weight.shape[1] // in_group_size
        num_codes_to_update = int(math.ceil(max_update_fraction * num_output_groups * num_input_groups))
        difference_with_reference_squared_norms = groupwise_squared_norms(
            reference_weight.grad, out_group_size, in_group_size)
        # ^-- [num_output_groups, num_input_groups]
        flat_indices_to_update = torch.topk(difference_with_reference_squared_norms.flatten(),
                                                k=num_codes_to_update, largest=True, sorted=True).indices
        
        return flat_indices_to_update


def groupwise_squared_norms(delta: torch.Tensor, out_group_size: int, in_group_size: int):
        """
        Given a matrix delta [out_features, in_features], compute a tensor [num_output_groups, num_input_groups] that
        contains the squared sum of elements of delta from each tile of (out_group_size, in_group_size) values.
        """
        return delta.view(delta.shape[0] // out_group_size, out_group_size,
                          delta.shape[1] // in_group_size, in_group_size).square().sum(dim=(1, 3))


def undo_repeat_interleave(repeated_tensor: torch.Tensor, repeats: int, dim=0):
    if dim == 0:
        if repeated_tensor.size(0) % repeats != 0:
            raise ValueError("Total number of rows is not divisible by repeats")
        reshaped_tensor = repeated_tensor.view(-1, repeats, repeated_tensor.size(1))
        return reshaped_tensor[:, 0, :]
    elif dim == 1:
        if repeated_tensor.size(1) % repeats != 0:
            raise ValueError("Total number of columns is not divisible by repeats")
        reshaped_tensor = repeated_tensor.view(repeated_tensor.size(0), -1, repeats)
        return reshaped_tensor[:, :, 0]
    else:
        raise ValueError("dim must be 0 or 1")


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


@torch.no_grad()
def pack_32bit_to_4bit(unpacked_weights, unpacked_qzeros):
    # Calculate the size of packed tensors
    packed_weight_shape = (unpacked_weights.shape[0] // 8, unpacked_weights.shape[1])
    packed_zeros_shape = (unpacked_qzeros.shape[0], unpacked_qzeros.shape[1] // 8)
    
    # Initialize packed tensors
    qweights = torch.zeros(
        packed_weight_shape,
        dtype=torch.int32,  # We'll use int32 to accommodate the packed 4-bit values
        device=unpacked_weights.device,
        requires_grad=False
    )
    
    qzeros = torch.zeros(
        packed_zeros_shape,
        dtype=torch.int32,  # We'll use int32 to accommodate the packed 4-bit values
        device=unpacked_qzeros.device,
        requires_grad=False
    )
    
    # Pack the weights
    for row in range(packed_weight_shape[0]):
        for i in range(8):
            qweights[row, :] |= (unpacked_weights[row * 8 + i, :] & 0xF) << (4 * i)
    
    # Pack the zeros
    for col in range(packed_zeros_shape[1]):
        for i in range(8):
            qzeros[:, col] |= (unpacked_qzeros[:, col * 8 + i] - 1 & 0xF) << (4 * i)
    
    return qweights, qzeros
