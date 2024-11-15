import torch

class ElasticEnergy:
    def __init__(self, young, poisson):
        '''
        Args:
            young: Young's modulus [Pa]
            poisson: Poisson ratio
        '''
        self.young = young
        self.poisson = poisson
        self.lbda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.mu = young / (2.0 * (1.0 + poisson))
        
    def compute_strain_tensor(self, jac):
        '''
        This method computes the strain tensor (#t, 3, 3)

        Args:
            jac: jacobian of the deformation (#t, 3, 3)
        
        Returns:
            E: strain induced by the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def compute_energy_density(self, jac, E):
        '''
        This method computes the energy density at each tetrahedron (#t,)

        Args:
            jac: jacobian of the deformation (#t, 3, 3)
            E: strain induced by the deformation (#t, 3, 3)
        
        Returns:
            psi: energy density per tet (#t,)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

    def compute_piola_kirchhoff_stress_tensor(self, jac, E):
        '''
        This method computes the stress tensor (#t, 3, 3)

        Args:
            jac: jacobian of the deformation (#t, 3, 3)
            E: strain induced by the deformation (#t, 3, 3)
        
        Returns:
            P: stress tensor induced by the deformation (#t, 3, 3)
        '''

        print("Please specify the kind of elasticity model.")
        raise NotImplementedError

class LinearElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)
    
    def compute_strain_tensor(self, jac):
        # 1/2 * (F^T * F) - I
        return 0.5 * (jac.transpose(-1, -2) + jac) - torch.eye(jac.shape[-1])

    def compute_energy_density(self, jac, E):
        # mu * E:E
        first = self.mu * torch.einsum('...ij,...ij', E, E)
        # 1/2 * lambda * (tr(E))^2
        second = 0.5 * self.lbda * torch.einsum('...ii->...', E)**2
        
        return first + second
    
    def compute_piola_kirchhoff_stress_tensor(self, jac, E):
        # 2 * mu * E
        first = 2 * self.mu * E
        # lambda * tr(E) 
        second = self.lbda * torch.einsum('...ii', E) 
        # lambda * tr(E) * I
        second = torch.einsum('..., ...ij -> ...ij', second, torch.eye(E.size(-1)))
        
        return first + second

class NeoHookeanElasticEnergy(ElasticEnergy):
    def __init__(self, young, poisson):
        super().__init__(young, poisson)
        self.muNH = self.mu
        self.lbdaNH = self.lbda + self.muNH
        self.nuNH = (self.lbdaNH - self.muNH) / (2.0 * self.lbdaNH)
        self.alpha = 1.0 + self.muNH / self.lbdaNH
        self.psi0 = 0.5 * self.lbdaNH * (self.alpha - 1)**2

    def compute_strain_tensor(self, jac):
        # Nothing to do here. Think why?
        # Neo-hookean model uses F directly, no strain needs to be calculated
        pass

    def compute_energy_density(self, jac, E):
        # I1 = tr(F^T * F)
        I1 = torch.einsum(
            '...ii',
            torch.bmm(jac, jac.transpose(-1, -2))
        )
        J = torch.det(jac)
        
        # psi1 = muNH/2 * (I1 - 3) 
        first = 0.5 * self.muNH * (I1 - 3)
        # psi2 = lbdaNH/2 * (J - alpha)^2
        second = 0.5 * self.lbdaNH * (J - self.alpha)**2

        return first + second - self.psi0

    def compute_piola_kirchhoff_stress_tensor(self, jac, E):
        # PK = muNH * F + lbdaNH * J * (J - alpha) * F^{-T}

        ## muNh * F
        first = self.muNH * jac

        ## lbdaNH * J * (J - alpha) * F^{-T}
        J = torch.det(jac)
        inv_jac = torch.inverse(jac)
        #       lbdaNH * J * (J - alpha) 
        second = self.lbdaNH * J * (J - self.alpha) 
        #       lbdaNH * J * (J - alpha) * F^{-T}
        second = torch.einsum("..., ...ij -> ...ij", second, inv_jac.transpose(-1, -2))

        return first + second
