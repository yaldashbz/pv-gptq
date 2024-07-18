import sys
import torch
import unittest
sys.path.append('../AQLM')

from src.aq_ops import IntCodes
from src.gptq_ops import unpack_4bit_to_32bit_signed, pack_32bit_to_4bit, undo_repeat_interleave


class TestGPTQOps(unittest.TestCase):
    def  test_pack_unpack(self):
        qweight = torch.randint(0, 15, (256, 5632))
        qzeros = torch.randint(0, 15, (16, 704))
        unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)
        packed_qweight, packed_qzeros = pack_32bit_to_4bit(unpacked_qweight, unpacked_qzeros)

        self.assertTrue(torch.equal(qweight, packed_qweight))
        self.assertTrue(torch.equal(qzeros, packed_qzeros))
        self.assertTrue(qweight.shape == packed_qweight.shape)
        self.assertTrue(qzeros.shape == packed_qzeros.shape)

    def test_undo_repeat_interleave(self):
        qweight = torch.randint(0, 15, (256, 5632))
        qzeros = torch.randint(0, 15, (16, 704))
        group_size = 128
        unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
        packed_qweight, packed_qzeros = pack_32bit_to_4bit(unpacked_qweight, unpacked_qzeros)
        packed_qzeros = undo_repeat_interleave(packed_qzeros, group_size, dim=0)

        self.assertTrue(torch.equal(qweight, packed_qweight))
        self.assertTrue(torch.equal(qzeros, packed_qzeros))
        self.assertTrue(qweight.shape == packed_qweight.shape)
        self.assertTrue(qzeros.shape == packed_qzeros.shape)

    def test_intcodes(self):
        pqweight = torch.randint(0, 15, (256, 5632))
        pqzeros = torch.randint(0, 15, (16, 704))
        qweight = IntCodes(pqweight)()
        qzeros = IntCodes(pqzeros)()
        assert torch.equal(pqzeros, qzeros)
        self.assertTrue(torch.equal(pqweight, qweight))
        group_size = 128
        unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(qweight, qzeros)
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
        packed_qweight, packed_qzeros = pack_32bit_to_4bit(unpacked_qweight, unpacked_qzeros)
        packed_qzeros = undo_repeat_interleave(packed_qzeros, group_size, dim=0)

        self.assertTrue(torch.equal(qweight, packed_qweight))
        self.assertTrue(torch.equal(qzeros, packed_qzeros))
        self.assertTrue(qweight.shape == packed_qweight.shape)
        self.assertTrue(qzeros.shape == packed_qzeros.shape)

    def test_quantlinear(self):
        prev_qweight = torch.randint(0, 15, (256, 5632))
        prev_qzeros = torch.randint(0, 15, (16, 704))
        w_grad = torch.rand(5632, 2048)
        new_scales = torch.rand(16, 5632)
        lr = 1e-1
        unpacked_qweight, unpacked_qzeros = unpack_4bit_to_32bit_signed(prev_qweight, prev_qzeros)
        group_size = unpacked_qweight.shape[0] // new_scales.shape[0]
        scales = new_scales.repeat_interleave(group_size, dim=0)
        unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)
        x = (unpacked_qweight - unpacked_qzeros) * scales
        print(unpacked_qweight.shape, unpacked_qzeros.shape, scales.shape, x.shape)
        qweight_grad = w_grad.T * scales
        qzeros_grad = -w_grad.T * scales

        qweight = unpacked_qweight - lr * qweight_grad
        qzeros = unpacked_qzeros - lr * qzeros_grad

        qweight, qzeros = qweight.round().int(), qzeros.round().int()
        qweight, qzeros = pack_32bit_to_4bit(qweight, qzeros)
        qzeros = undo_repeat_interleave(qzeros, group_size, dim=0)

        qweight_change_rate = torch.not_equal(prev_qweight, qweight).any(-1).float().mean().item()
        qzeros_change_rate = torch.not_equal(prev_qzeros, qzeros).any(-1).float().mean().item()

        print(qweight_change_rate, qzeros_change_rate)

        self.assertTrue(qweight.shape == prev_qweight.shape)
        self.assertTrue(qzeros.shape == prev_qzeros.shape)


if __name__ == '__main__':
    unittest.main()