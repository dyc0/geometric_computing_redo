import pytest

import json
import pytest
import torch

import sys
import os
sys.path.append('assignment_2_2/src')
TEST_PATH = "assignment_2_2/test"

import fem_system as fs


class TestLinearElasticEnergy:
    # def setup_method(self):
    #     self.LEE = ee.LinearElasticEnergy(1.0, 0.25)
    #     with open(
    #             os.path.join(TEST_PATH, "elastic_energy_data.json"), "r"
    #         ) as json_file:
    #         self.data = json.load(json_file)
    
    @pytest.mark.timeout(1)
    def test_barycenters(self):
        v = torch.tensor([
            [0.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]])
        tet = torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 2, 4]
        ])
        
        barycenters = fs.compute_barycenters(v, tet)
        solution = torch.tensor([
            [0.25, 0.25, 0.25], 
            [0.25, 0.5,  0.5]])
        assert torch.allclose(barycenters, solution, atol=1e-10)

    @pytest.mark.timeout(1)
    def test_shape_matrices(self):
        v = torch.tensor([
            [0.5, 0.5, 0.0], 
            [0.0, 0.0, 1.0], 
            [0.0, 1.0, 0.0], 
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        tet = torch.tensor([
            [4, 3, 2, 1],
            [4, 3, 2, 0]
        ])

        D = fs.compute_shape_matrices(v, tet)
        solution = torch.tensor([
            [[1.0,  1.0,  0.0],
             [1.0,  0.0,  1.0],
             [0.0, -1.0, -1.0]],
            
            [[0.5,  0.5, -0.5],
             [0.5, -0.5,  0.5],
             [1.0,  0.0,  0.0]]
        ])

        assert D.shape == (2, 3, 3)
        assert torch.allclose(D, solution)

    @pytest.mark.timeout(1)
    def test_pin_mask(self):
        nv = 32
        v = torch.zeros((nv, 3))
        tet = torch.tensor([])
        pin_idx = [0, 1, 2, 7, 8, 10, 23]

        free_indices = torch.tensor([i for i in range(32) if i not in pin_idx])
        free_mask = torch.ones((nv, 1))
        free_mask[pin_idx] = 0

        # test pins existing
        fem_obj = fs.FEMSystem(v, tet, pin_idx=pin_idx)
        assert fem_obj.free_idx.shape == torch.Size([nv - len(pin_idx)])
        assert fem_obj.free_mask.shape == torch.Size([nv, 1])
        assert torch.allclose(fem_obj.free_mask, free_mask, atol=1e-10)
        assert torch.allclose(fem_obj.free_idx, free_indices, atol=1e-10)

