import torch

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
    
    def __init__(self, v_rest, tet, rho=1.0, pin_idx=None):
        '''
        Args:
            v_rest: rest position of the vertices of the mesh (#v, 3)
            tet: indices of the element's vertices (#t, 4)
            rho: mass per unit volume [kg.m-3]
            pin_idx: list of vertex indices to pin
        '''

        self.v_rest = v_rest.clone()
        self.tet = tet.clone()
        self.rho = rho
        if pin_idx is None:
            pin_idx = []
        self.pin_idx = torch.tensor(pin_idx)
        self.free_idx = None
        self.free_mask = None
        self.make_free_indices_and_free_mask()

        self.rest_barycenters = None
        self.Dm = None
        self.Bm = None
        self.W0 = None
        
        self.update_rest_shape(self.v_rest)
        
    def make_free_indices_and_free_mask(self):
        '''
        Updated attributes:
            free_idx: torch tensor of shape (#free_vertices,) containing the list of unpinned vertices
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
        # TODO: Write a test, though I'm fairly certain it's correct
        # pin vertices that shouldn't be changed
        v_def_pinned = self.compute_pinned_deformation(v_def)

        # compute deformed shape matrix
        Ds = compute_shape_matrices(v_def_pinned, self.tet)

        return torch.bmm(Ds, self.Bm)

def compute_barycenters(v, tet):
    '''
    Args:
        v: vertices of the mesh (#v, 3)
        tet: indices of the element's vertices (#t, 4)
        
    Returns:
        barycenters: barycenters of the tetrahedra (#t, 3)
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
