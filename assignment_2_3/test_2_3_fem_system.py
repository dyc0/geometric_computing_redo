import json
import pytest
import torch

import sys
import os
sys.path.append('assignment_2_3/src')
TEST_PATH = "assignment_2_3/test"

import elastic_energy as ee
import fem_system as fs

class TestFemSystem:
    def setup_method(self):
        elasticity = ee.NeoHookeanElasticEnergy(1.0, 0.25)
        with open(
                os.path.join(TEST_PATH, "neohookeandata.json"), "r"
            ) as json_file:
            self.ee_data = json.load(json_file)
        with open(
                os.path.join(TEST_PATH, "fem_data.json"), "r"
            ) as json_file:
            self.fem_data = json.load(json_file)

        self.fem = fs.FEMSystem(
            torch.tensor(self.fem_data["vertices"], dtype=torch.float64),
            torch.tensor(self.fem_data["tets"], dtype=torch.long),
            elasticity,
            f_mass = torch.tensor([0, 0, -9.8], dtype=torch.float64)
        )
    
    @pytest.mark.timeout(5)
    def test_forces(self):
        # TODO: Sometimes fails with
        # Test result not found for: ./assignment_2_3/test_2_3_fem_system.py::TestFemSystem::test_forces
        # Also, sometimes fails with
        # TypeError: __init__() got an unexpected keyword argument 'f_mass'
        jac = torch.tensor(self.ee_data["jac_tensor"], dtype=torch.float64)
        F = self.fem.compute_elastic_forces(jac, None)

        assert F.dtype == torch.float64
        assert torch.allclose(
            F, 
            torch.tensor(self.fem_data["f_el"], dtype=torch.float64),
            )
        
    @pytest.mark.timeout(1)
    def test_external_forces(self):
        self.fem.compute_volumetric_and_external_forces()
        
