import sys
import math
import torch
import unittest
sys.path.append('../AQLM')


from src.gptq_ops import QuantLinear, groupwise_squared_norms, get_discrete_group_indices
from src.aq_ops import IntCodes

class TestQuantLinear(unittest.TestCase):
    method = QuantLinear()

    def test_discrete_group_update(self):
        reference_weight = torch.ones(3,6)
        targeted_tensor = reference_weight.t()
        reference_weight_grad = torch.tensor([[7,8,9,10,11,12], [13,14,15,16,17,18], [1,2,3,4, 5, 6]]).float()
        reference_weight.grad = reference_weight_grad
        targeted_tensor_grad = torch.rand(targeted_tensor.shape)
        group_size = 2
        lr = 100
        k = 4
        difference_with_reference_squared_norms = groupwise_squared_norms(reference_weight_grad, 1, group_size)
        flat_indices_to_update = torch.topk(difference_with_reference_squared_norms.flatten(),
                                                k=k, largest=True, sorted=True).indices
        flat_indices_to_update = get_discrete_group_indices(reference_weight, 1, group_size, 0.2)
        print(flat_indices_to_update)
        num_groups = reference_weight.shape[1] // group_size

        # Calculate row and group positions in w
        row_indices = flat_indices_to_update // num_groups
        group_positions = flat_indices_to_update % num_groups

        # Calculate the start and end positions in q
        start_cols = group_positions * group_size
        end_cols = start_cols + group_size

        for row, start_col, end_col in zip(row_indices, start_cols, end_cols):
            targeted_tensor[start_col:end_col, row] = targeted_tensor[start_col:end_col, row] - lr * targeted_tensor_grad[start_col:end_col, row]
            targeted_tensor[start_col:end_col, row].round().int()

        print(targeted_tensor)
    
    def test_QuantLinear_real_tensors(self):
        loaded_tensors = torch.load('tests/test_tensors.pth')
        prev_qweight = loaded_tensors['qweight']
        prev_qzeros = loaded_tensors['qzeros']
        scales = loaded_tensors['scales']
        reference_weight = loaded_tensors['reference_weight']
        reference_weight.grad = reference_weight.clone()

        new_qweight, new_qzeros = self.method.update_discretes(prev_qweight, prev_qzeros, scales, reference_weight, 0.01, 0)
        self.assertTrue(torch.equal(prev_qweight, new_qweight))
        self.assertTrue(torch.equal(prev_qzeros, new_qzeros))



if __name__ == '__main__':
    unittest.main()