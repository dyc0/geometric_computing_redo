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
            rho = 0.7,
            f_mass = torch.tensor([-1, 2, -3], dtype=torch.float64)
        )
    
    @pytest.mark.timeout(5)
    def test_forces(self):
        # NOTE: Sometimes fails with
        # Test result not found for: ./assignment_2_3/test_2_3_fem_system.py::TestFemSystem::test_forces
        # Also, sometimes fails with
        # TypeError: __init__() got an unexpected keyword argument 'f_mass'
        # I think it's a bug unrelated to our code, as it has already happened to me before,
        # and it wanished when I reloaded Docker
        jac = torch.tensor(self.ee_data["jac_tensor"], dtype=torch.float64)
        F = self.fem.compute_elastic_forces(jac, None)

        assert F.dtype == torch.float64
        assert torch.allclose(
            F, 
            torch.tensor(self.fem_data["f_el"], dtype=torch.float64),
            )
        
    @pytest.mark.timeout(1)
    def test_external_forces(self):
        f_vol, f_ext = self.fem.compute_volumetric_and_external_forces()

        f_vol_correct = torch.tensor(self.fem_data["f_vol"], dtype=torch.float64)
        f_ext_correct = torch.tensor(self.fem_data["f_ext"], dtype=torch.float64)

        assert f_vol.dtype == torch.float64
        assert f_ext.dtype == torch.float64

        assert torch.allclose(f_vol, f_vol_correct)
        assert torch.allclose(f_ext, f_ext_correct)

    @pytest.mark.timeout(1)
    def test_external_energy(self):
        def_barycenters = torch.tensor(self.fem_data["def_barycenters"], dtype=torch.float64)
        f_vol, _ = self.fem.compute_volumetric_and_external_forces()
        E = self.fem.compute_external_energy(def_barycenters, f_vol)
        E_correct = torch.tensor(self.fem_data["E"], dtype=torch.float64)

        assert E.dtype == torch.float64
        assert torch.allclose(E, E_correct)
        
