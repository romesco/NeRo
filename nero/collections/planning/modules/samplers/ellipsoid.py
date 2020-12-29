import torch
import pyro

from typing import List, Tuple

class EllipsoidSampler:
    """
    Implements the direct transformation-based technique shown by Gammell et al.
    https://arxiv.org/pdf/1706.06454%22%3e1706.06454.pdf pg. 9 Algs 1 & 2

    """
    def __init__(
        self,
        focus_start_list: List[Tuple[float]],
        focus_goal_list: List[Tuple[float]],
        current_solution_cost: int,
        major_axis_lengths: List[float],
        min_bounds: List[float],
        max_bounds: List[float],
        ):

        focus_start_tensor = torch.Tensor(focus_start_list)
        focus_goal_tensor = torch.Tensor(focus_goal_list)

        self.dim = len(focus_start_list[0]) # num items in tuple
        self.center = (focus_start_tensor + focus_goal_tensor).sum(dim=0)/2
        self.c_i = torch.Tensor([current_solution_cost])

        # c_min = ||x_goal - x_start||_2
        self.c_min = torch.linalg.norm((focus_goal_tensor - focus_start_tensor), ord=2)
        a_1 = (focus_goal_tensor - focus_start_tensor) / self.c_min

        # For problems seeking to minimize path length:
        # M = a_1 * 1_1^T 
        M = a_1 * torch.eye(self.dim)[0,None].T
        U, S, V = torch.svd(M)
        
        # /Lambda = diag(1,...1, det(U)*det(V))
        Lambda = torch.eye(self.dim)
        Lambda[-1,-1] = torch.linalg.det(U) * torch.linalg.det(V)

        # C_we = U * /Lambda * V^T
        self.C_we = U @ Lambda @ V.T
        
    def sample(self):

        r_1 = self.c_i / 2
        r_list = [r_1]
        for j in range(2,self.dim+1):
            r_list.append(torch.sqrt(self.c_i**2 - self.c_min**2)/2)

        L = torch.diag(torch.Tensor(r_list))
            
        # Sample from unit hypersphere
        unit_sphere_sample = pyro.sample("unit_sphere", pyro.distributions.Uniform(torch.zeros(self.dim), 1))

        # Transform into hyperspheroid
        ellipsoid_sample = self.C_we @ L @ unit_sphere_sample * self.center
         
        return ellipsoid_sample

    def reject(self):
        pass
        
if __name__ == '__main__':
    sampler = EllipsoidSampler([(1,1)], [(2,2)], 5, [1], [0], [5])
    for i in range(10):
        print(sampler.sample())


