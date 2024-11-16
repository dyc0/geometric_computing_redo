import json
import pytest
import torch

import sys
import os
sys.path.append('assignment_2_3/src')
TEST_PATH = "assignment_2_3/test"

import elastic_energy as ee


class TestLinearElasticEnergy:
    def setup_method(self):
        self.LEE = ee.LinearElasticEnergy(1.0, 0.25)
        with open(
                os.path.join(TEST_PATH, "elastic_energy_data.json"), "r"
            ) as json_file:
            self.data = json.load(json_file)
    
    @pytest.mark.timeout(1)
    def test_strain_tensor(self):
        F = torch.tensor(self.data["jac_tensor"], dtype=torch.float64)
        E = self.LEE.compute_strain_tensor(F)
        E_hand = torch.tensor(self.data["linear_strain_tensor"], dtype=torch.float64)
        assert E.dtype == torch.float64
        assert E.shape == F.shape
        assert torch.allclose(E, E_hand, atol=1e-5)        

    @pytest.mark.timeout(1)
    def test_energy_density(self):
        F = torch.tensor(self.data["jac_tensor"], dtype=torch.float64)
        E = self.LEE.compute_strain_tensor(F)
        psi = self.LEE.compute_energy_density(F, E)
        psi_hand = torch.tensor(self.data["psi_hand"], dtype=torch.float64)
        assert psi.dtype == torch.float64
        assert psi.shape == torch.Size([3])
        assert torch.allclose(psi, psi_hand, atol=1e-5)
        
    @pytest.mark.timeout(1)
    def test_piola_kirchhoff_stress_tensor(self):
        F = torch.tensor(self.data["jac_tensor"], dtype=torch.float64)
        E = self.LEE.compute_strain_tensor(F)
        P = self.LEE.compute_piola_kirchhoff_stress_tensor(F, E)
        P_hand = torch.tensor(self.data["pk1_hand"], dtype=torch.float64)
        assert P.dtype == torch.float64
        assert P.shape == F.shape
        assert torch.allclose(P, P_hand, atol=1e-5)


class TestNeoHookeanElasticEnergy:
    def setup_method(self):
        self.NHEE = ee.NeoHookeanElasticEnergy(1.0, 0.25)
        with open(
                os.path.join(TEST_PATH, "neohookeandata.json"), "r"
            ) as json_file:
            self.data = json.load(json_file)
         

    def test_energy_density(self):
        F = torch.tensor(self.data["jac_tensor"], dtype=torch.float64)
        E = self.NHEE.compute_strain_tensor(F)
        psi = self.NHEE.compute_energy_density(F, E)
        psi_hand = torch.tensor(self.data["psi_hand"], dtype=torch.float64)
        assert psi.dtype == torch.float64
        assert psi.shape == psi_hand.shape 
        assert torch.allclose(psi, psi_hand, atol=1e-5)
        

    def test_piola_kirchhoff_stress_tensor(self):
        F = torch.tensor(self.data["jac_tensor"], dtype=torch.float64)
        E = self.NHEE.compute_strain_tensor(F)
        P = self.NHEE.compute_piola_kirchhoff_stress_tensor(F, E)
        P_hand = torch.tensor(self.data["pk1_hand"], dtype=torch.float64)
        assert P.dtype == torch.float64
        assert P.shape == F.shape
        assert torch.allclose(P, P_hand, atol=1e-5)