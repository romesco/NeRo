import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import pyro
import pyro.distributions as dist

def plot_bandits(dist_params_history):
    """
    dist_params_history: 
    dim0: num_bandits
    dim1: num_rounds
    dim2: num_params
    """
    num_rounds = dist_params_history.shape[0]
    num_bandits = dist_params_history.shape[1]
    num_params = dist_params_history.shape[2]
    fig = plt.figure()
    support = torch.arange(-0.1, 1.1, 0.001).detach()
    
    max_y = 0
    for b in range(num_bandits):
        print(b)
        d = dist.Beta(*dist_params_history[0][b])
        prob = d.log_prob(support).detach().exp()
        plt.plot(support.numpy(), prob.numpy())
        max_yb = prob.numpy()[~np.isnan(prob.numpy())].max()
        if max_yb > max_y:
            max_y = max_yb
        


    plt.xlim(-0.1, 1.1)
    plt.ylim(0, max_y*1.05) 
    plt.xlabel('support')
    plt.ylabel('p')
    plt.title('Beta function')
    plt.tight_layout();
    plt.show()
