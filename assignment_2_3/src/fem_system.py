import torch
from vertex_elt_sum import VertexEltsSum

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#                             FEM SYSTEM CLASS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

class FEMSystem():
    
    def __init__(self, v_rest, tet, ee, rho=1.0, pin_idx=None, f_mass=None):
        '''
        Args:
            v_rest: rest position of the vertices of the mesh (#v, 3)
            tet: indices of the element's vertices (#t, 4)
            ee: ElasticEnergy object that can be found in elastic_energy.py
            rho: mass per unit volume [kg.m-3]
            pin_idx: list of vertex indices to pin
            f_mass: torch tensor, external force per unit mass (3,) [N.kg-1]
        '''

        self.v_rest = v_rest.clone()
        self.tet = tet.clone()
        self.ee = ee
        self.rho = rho
        if pin_idx is None:
            pin_idx = []
        self.pin_idx = torch.tensor(pin_idx)
        self.free_idx = None
        self.free_mask = None
        self.make_free_indices_and_free_mask()
        self.f_mass = f_mass.clone()

        self.rest_barycenters = None
        self.Dm = None
        self.Bm = None
        self.W0 = None
        
        self.update_rest_shape(self.v_rest)

        self.vertex_tet_sum = VertexEltsSum(self.v_rest.shape[0], self.tet)
        
    def make_free_indices_and_free_mask(self):
        '''
        Updated attributes:
            free_index: torch tensor of shape (#free_vertices,) containing the list of unpinned vertices
            free_mask: torch tensor of shape (#v, 1) containing 1 at free vertex indices and 0 at pinned vertex indices
        '''
        # They are evil and this is pathological
        self.pin_idx = self.pin_idx.long()

        self.free_mask = torch.ones(self.v_rest.shape[0])
        if not self.pin_idx.numel() == 0:
            self.free_mask[self.pin_idx] = 0

        self.free_idx = torch.nonzero(self.free_mask).squeeze()
        self.free_mask = torch.unsqueeze(self.free_mask, -1)
        
    def update_rest_shape(self, v_rest):
        '''
        Args:
            v_rest: rest position of the vertices of the mesh (#v, 3)
        
        Updated attributes:
            v_rest: rest position of the vertices of the mesh (#v, 3)
            rest_barycenters: barycenters of the tetrahedra (#t, 3)
            Dm: shape matrices of the undeformed configuration (#t, 3, 3)
            Bm: inverse of the shape matrices of the undeformed configuration (#t, 3, 3)
            W0: signed volumes of the tetrahedra (#t,)
        '''
        self.v_rest = v_rest.clone()
        self.rest_barycenters = compute_barycenters(self.v_rest, self.tet)
        self.Dm = compute_shape_matrices(self.v_rest, self.tet)
        # Compute batch inverse, is alias for torch.linalg.inv
        self.Bm = torch.inverse(self.Dm)
        self.W0 = compute_signed_volume(self.Dm)

    def compute_pinned_deformation(self, v_def):
        '''
        Args:
            v_def: deformed position of the vertices of the mesh (#v, 3)

        Returns:
            v_def_pinned: deformed position of the vertices of the mesh after taking pinning into account (#v, 3)
        '''
        v_def_pinned = v_def.clone()
        v_def_pinned[self.pin_idx] = self.v_rest[self.pin_idx]

        return v_def_pinned
        
    def compute_jacobians(self, v_def):
        '''
        Args:
            v_def: deformed position of the vertices of the mesh (#v, 3)

        Returns:
            jac: Jacobians of the deformation (#t, 3, 3)
        '''
        v_def_pinned = self.compute_pinned_deformation(v_def)

        # compute deformed shape matrix
        Ds = compute_shape_matrices(v_def_pinned, self.tet)

        return torch.bmm(Ds, self.Bm)
    
    def compute_strain_tensor(self, jac):
        '''
        Args:
            jac: Jacobians of the deformation (#t, 3, 3)

        Returns:
            E: strain induced by the deformation (#t, 3, 3)
        '''
        pass
    
    ## Energies ##
    
    def compute_elastic_energy(self, jac, E):
        '''
        Args:
            jac: Jacobians of the deformation (#t, 3, 3)
            E: strain induced by the deformation (#t, 3, 3)
        
        Returns:
            energy_el: elastic energy of the system [J]
        '''
        
    
    def compute_external_energy(self, def_barycenters, f_vol):
        '''
        This computes the external energy potential
        
        Args:
            def_barycenters: barycenters of the deformed tetrahedra (#t, 3)
            f_vol: external force per unit volume acting on the tets (#t, 3)

        Returns:
            energy_ext: postential energy due to external forces [J]
        '''
        pass
    
    ## Forces ##
    
    def compute_elastic_forces(self, jac, E):
        '''
        Args:
            jac: the jacobian of the deformation (#t, 3, 3)
            E: strain induced by the deformation (#t, 3, 3), can be None for Neo-Hookean
            
        Returns:
            forces_el: the elastic forces of the system (#v, 3)
        '''
        pass
    
    def compute_volumetric_and_external_forces(self):
        '''
        Convert force per unit mass to volumetric forces (in the undeformed state), then distribute
        the forces to the vertices of the mesh.

        Returns:
            f_vol: torch tensor of shape (#t, 3) external force per unit volume acting on the tets
            f_ext: torch tensor of shape (#v, 3) external force acting on the vertices
        '''
        pass
    

def compute_barycenters(v, tet):
    '''
    Args:
        v: vertices of the mesh (#v, 3)
        tet: indices of the element's vertices (#t, 4)
        
    Returns:
        barycenters: barycenters of the tetrahedra
    '''
    # Mean is along the same spatial axis in each tetrahedron
    return torch.mean(v[tet], axis=1)

def compute_shape_matrices(v, tet):
    '''
    Args:
        v: vertices of the mesh (#v, 3)
        tet: indices of the element's vertices (#t, 4)

    Returns:
        D: shape matrices of current configuration (#t, 3, 3)
    '''
    tet_v = v[tet]
    # -1: is needed for dimensions to match
    D = tet_v[:, :-1, :] - tet_v[:, -1:, :]
    return D.mT

def compute_signed_volume(D):
    '''
    Args:
        D: shape matrices of current configuration (#t, 3, 3)

    Returns:
        signed_volume: signed volumes of the tetrahedra (#t,)
    '''
    # torch.det does batch determinant
    return -torch.det(D) / 6.0
