import numpy as np
import torch
from torch.func import vmap, grad, functional_call
import torch.func as func
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm, trange
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from neural_network_classes import ACnet
from neural_network_classes import SimpleNN

torch.set_default_device('cuda')

def sample_unit_ball(n_dims, n_samples): #samples uniformly (i.e. by volume)
    gaussian_points = torch.randn(n_samples, n_dims)
    sphere_points = gaussian_points / torch.linalg.norm(gaussian_points, dim=1, keepdim=True)
    radii = torch.rand(n_samples, 1) ** (1/n_dims)
    return sphere_points * radii

def sample_hypercube(n_dims, n_samples):
    return 2 * torch.rand(n_samples, n_dims) - 1

dim = 10 #dimension of state space/domain Omega
dim_ac = 1 #dimension of the action space A
dim_bm = dim #dimension of the Brownian motion W_t driving the SDE
N_act = 2**9
N_cri = 2**9
beta_net = 0.6

# HJB equation parameters
gamma = 1
R = 1

# Initialization
qnet = ACnet(dim, N_cri, 1, beta_net)
#qnet = SimpleNN(dim_in = dim, num_neurons = 50, num_layers = 2, dim_out = 1)
qparams = dict(qnet.named_parameters())

unet = ACnet(dim, N_act, dim_ac, beta_net)
#unet = SimpleNN(dim_in = dim, num_neurons = 50, num_layers = 2, dim_out = dim_ac)
uparams = dict(unet.named_parameters())

#Hyperparameter functions and their vectorized counterparts
def eta_point(point):
    return torch.prod(1-point**2)
def f(point): #extrapolated boundary condition
    return 1
def critic_point_func(qnet, qparams, point): #should evaluate whether all this sequeezing is actually needed
    qnet_output = functional_call(qnet, qparams, (point.unsqueeze(0),))
    scalar_out = qnet_output.squeeze()  # shape []
    eta = eta_point(point)
    return eta * scalar_out + f(point)
def actor_point_func(unet, uparams, point): #could be more complicated for action spaces that are not \R^{dim_ac}
    unet_output = functional_call(unet, uparams, (point.unsqueeze(0),))
    return unet_output.squeeze()  # shape []

critic_vec = vmap(critic_point_func, in_dims = (None, None, 0))
actor_vec = vmap(actor_point_func, in_dims = (None, None, 0))

def true_critic(point): #must be a function of point, even if trivially
    return 1 + torch.sum(point ** 2) * torch.prod(torch.sin(np.pi * point))
def true_actor(point): #must be a function of point, even if trivially
    return point[0] * torch.sin(np.pi * point[2]) + point[1]

true_critic_vec = vmap(true_critic)
true_actor_vec = vmap(true_actor)

#HJB coefficients
def drift(point, action): #outputs dim-dimensional vector
    return action * point
def diffusion(point, action): #outputs dim x dim_bm matrix
    #output_vec = (1 + action**2 + point**2)
    #return torch.reshape(output_vec, (-1,dim_bm))
    return torch.eye(dim, dim_bm)
def zeta(point, action): #outputs dim_ac-dimensional vector
    return (action - true_actor(point)) ** 2

true_hess, true_grad = func.hessian(true_critic), func.grad(true_critic)
def cost(point, action):
    dif = diffusion(point, action)
    diff_term = torch.matmul(dif, dif.transpose(0,1)) * true_hess(point)
    return zeta(point, action) + gamma * true_critic(point) - drift(point, action).dot(true_grad(point)) - diff_term.sum() / 2

drift_vec, diffusion_vec, zeta_vec, cost_vec = vmap(drift), vmap(diffusion), vmap(zeta), vmap(cost)

critic_hess, critic_grad = func.hessian(critic_point_func, argnums=2), func.grad(critic_point_func, argnums=2)
critic_hess_vec, critic_grad_vec = vmap(critic_hess, in_dims = (None, None, 0)), vmap(critic_grad, in_dims = (None, None, 0))

def sec_order_mult(diffusion, hessian):
    return (torch.matmul(diffusion, diffusion.transpose(0, 1)) * hessian).sum() / 2
def first_order_mult(drift, gradient):
    return drift.dot(gradient)
sec_order_mult_vec, first_order_mult_vec = vmap(sec_order_mult), vmap(first_order_mult)

start = 0 #0 -> train actor first, 1 -> train critic first
Num_iterate = 4000
Nbasepoints = int(1e3) #Used to evaluate loss values for graphs (plays no role in training)
Nmc = 10000 #Number of points to use for evaluating integrals in each training cycle
Num_epo_cri = 1
Num_epo_act = 1
initial_lr_critic = 0.05
initial_lr_actor = 0.05
max_grad = 1000000
big_critic_loss, big_actor_loss = [], []
big_critic_op_loss, big_actor_ham_loss = [], []
big_rel_critic_loss, big_rel_actor_loss = [], []

#Numerical tricks
DGM = False
hamiltonian_optimizer = "normal" #Options are "normal", "ReLU", "leaky_ReLU", "abs_penalty", and "quadratic_penalty"
delta = -1 #Parameter for the above options -- should be nonpositive

#Optimzers and schedulers
Qoptimizer = optim.Adam(qnet.parameters(), lr=initial_lr_critic)
Qscheduler = LambdaLR(Qoptimizer, lr_lambda=lambda epoch: initial_lr_critic * (epoch // Num_epo_cri + 1) ** (-0.8))
Uoptimizer = optim.Adam(unet.parameters(), lr=initial_lr_actor)
Uscheduler = LambdaLR(Uoptimizer, lr_lambda=lambda epoch: initial_lr_actor * (epoch // Num_epo_act + 1) ** (-0.6))

for j in range(start, start + Num_iterate):
    source = R * sample_hypercube(dim, Nbasepoints)

    # Critic step:
    if (j%2):
        # Critic training
        for count1 in tqdm(range(Num_epo_cri)):
            grid = R * sample_hypercube(dim, Nmc)
            grid.requires_grad = True

            #Batch calculate actor and critic
            out = critic_vec(qnet, qparams, grid)
            uout = actor_vec(unet, uparams, grid).detach()

            #Calcaulate components of PDE operator
            dif = diffusion_vec(grid, uout)
            dri = drift_vec(grid, uout)
            cos = cost_vec(grid, uout)
            cri_grad = critic_grad_vec(qnet, qparams, grid)
            cri_hess = critic_hess_vec(qnet, qparams, grid)
            sec_order = sec_order_mult_vec(dif, cri_hess)
            first_order = first_order_mult_vec(dri, cri_grad)

            # PDE operator -- may use DGM or Q-PDE for critic loss
            op = sec_order + first_order + cos - gamma * out
            if DGM:
                loss_to_min = torch.dot(op, op) / Nmc
            else:
                lq = op.detach()
                loss_to_min = torch.dot(-lq, out) / Nmc

            # Critic update
            Qoptimizer.zero_grad()
            loss_to_min.backward()
            Qoptimizer.step()
            Qscheduler.step()

            # memory optimization (more to do)
            loss_to_min = loss_to_min.detach()
            op = op.detach()
            out = out.detach()
            uout = uout.detach()

            #lazy functional programming, but it works
            qparams = dict(qnet.named_parameters())

        # Comparison against true value function
        with torch.no_grad():
            critic_estimate = critic_vec(qnet, qparams, source)
            critic_true = true_critic_vec(source)
            critic_loss = torch.mean((critic_estimate - critic_true)**2).detach().cpu().numpy()
            rel_critic_loss = (torch.sum((critic_estimate - critic_true)**2) / torch.sum(critic_true**2)).detach().cpu().numpy()
            big_critic_loss.append(float(critic_loss))
            big_rel_critic_loss.append(float(rel_critic_loss)) ## Easier to graph floats than tensors
            big_critic_op_loss.append(float(loss_to_min))

    # Actor step:
    if not (j%2):
        # Actor training
        for count2 in tqdm(range(Num_epo_act)):
            grid = R * sample_hypercube(dim, Nmc)
            grid.requires_grad = True

            #Batch calculate actor and critic
            out = critic_vec(qnet, qparams, grid).detach()
            uout = actor_vec(unet, uparams, grid)

            #Calculate components of Hamiltonian
            dif = diffusion_vec(grid, uout)
            dri = drift_vec(grid, uout)
            cos = cost_vec(grid, uout)
            cri_grad = critic_grad_vec(qnet, qparams, grid)
            cri_hess = critic_hess_vec(qnet, qparams, grid)
            sec_order = sec_order_mult_vec(dif, cri_hess)
            first_order = first_order_mult_vec(dri, cri_grad)

            #Integral of Hamiltonian -- note that this leaves in unnecessary gradients
            ham = sec_order + first_order + cos - gamma * out
            if hamiltonian_optimizer == "normal":
                ih = ham.mean()
            elif hamiltonian_optimizer == "ReLU":
                ih = torch.clamp(ham, min = delta).mean()
            elif hamiltonian_optimizer == "leaky_ReLU":
                ih = (torch.clamp(ham, min = delta) + 0.1 * torch.clamp(ham, max = delta)).mean()
            elif hamiltonian_optimizer == "abs_penalty":
                ih = (torch.clamp(ham, min = delta) - 0.1 * torch.clamp(ham, max = delta)).mean()
            elif hamiltonian_optimizer == "quadratic_penalty":
                ih = (torch.clamp(ham, min = delta) + 0.1 * (torch.clamp(ham, max = delta) - delta) ** 2).mean()

            # Critic update
            Uoptimizer.zero_grad()
            ih.backward()
            Uoptimizer.step()
            Uscheduler.step()

            # memory optimization (more to do in preventing unneeded gradients from being calculated)
            ih = ih.detach()
            out = out.detach()
            uout = uout.detach()

            # lazy functional programming, but it works
            uparams = dict(unet.named_parameters())

        # Comparison against true control
        with torch.no_grad():
            actor_estimate = actor_vec(unet, uparams, source)
            actor_true = true_actor_vec(source)
            actor_loss = torch.mean((actor_estimate - actor_true)**2 * dim_ac).detach().cpu().numpy()
            rel_actor_loss = (torch.sum((actor_estimate - actor_true)**2) / torch.sum(actor_true**2)).detach().cpu().numpy()
            big_actor_loss.append(float(actor_loss))
            big_rel_actor_loss.append(float(rel_actor_loss))
            big_actor_ham_loss.append(float(ih))

    if ((j % 5 == 0) & (j > 2)):
        print(f"Iteration {j + 1 - start}")
        print(f"Critic mean square error: {critic_loss}")
        print(f"Critic relative error: {float(rel_critic_loss)}")
        print(f"Critic operator loss: {float(loss_to_min)}")
        print(f"Actor mean square error: {actor_loss}")
        print(f"Actor relative error: {float(rel_actor_loss)}")
        print(f"Actor operator loss: {float(ih)}")


#benchmarking
num_sims = 2000 #number of Monte Carlo samples of value function
hard_cut = 10000 #end Monte Carlo simulation at this point even if a particle has yet to exit domain
time_increment = 10 ** -3
matmul_vec = vmap(torch.matmul)
randn_vec = vmap(torch.randn)

def mc_benchmark(start_point, num_sims, time_increment, uparams, unet):
    starts = start_point.squeeze().repeat(num_sims, 1)
    running_costs = torch.zeros(num_sims)
    active_particles = torch.ones(num_sims, dtype = torch.bool)

    with torch.no_grad():
        for iteration in range(1, hard_cut):
            active_idx = active_particles.nonzero().squeeze()
            if active_particles.sum() <= 1: #workaround for bug that breaks vectorization if there is only one particle
                if active_particles.sum() == 0:
                    break
                ind = torch.nonzero(active_particles).item()
                for iteration_2 in range(iteration, hard_cut):
                    discount = math.exp(-gamma * iteration_2 * time_increment)
                    control = actor_point_func(unet, uparams, starts[ind])
                    #control = true_actor(starts[ind])

                    running_costs[ind] += time_increment * cost(starts[ind], control)
                    starts[ind] += time_increment * drift(starts[ind], control) + math.sqrt(time_increment) * torch.matmul(diffusion(starts[ind], control), torch.randn(dim_bm))

                    #Problem specific
                    if torch.abs(starts[ind]).max() >= R:
                        #Problem specific
                        running_costs[ind] += discount * 1
                        break
                break
            else:
                discount = math.exp(-gamma * iteration * time_increment)
                active_starts = starts[active_idx]
                control = actor_vec(unet, uparams, active_starts)
                #control = true_actor_vec(active_starts)

                running_costs[active_idx] += discount * cost_vec(active_starts, control) * time_increment
                starts[active_idx] += time_increment * drift_vec(active_starts, control) + math.sqrt(time_increment) * matmul_vec(diffusion_vec(active_starts, control), torch.randn(active_particles.sum(), dim_bm))

                #Problem specific
                norms = torch.abs(starts[active_idx]).max()
                escaped = norms >= R

                if escaped.any():
                    escaped_idx = active_idx[escaped]

                    #Problem specific
                    running_costs[escaped_idx] += discount * 1
                    active_particles[escaped_idx] = False

    return running_costs.mean()

#For loops are unfortunate but necessary here
num_samples = 1000
error_true_to_critic = torch.zeros(num_samples)
error_true_to_mc = torch.zeros(num_samples)
error_critic_to_mc = torch.zeros(num_samples)

for i in range(1,num_samples):
    point = sample_unit_ball(dim, 1)
    mc = mc_benchmark(point, num_sims, time_increment, uparams, unet)
    critic = critic_point_func(qnet, qparams, point)
    true_cri = true_critic(point)

    error_critic_to_mc[i] = critic - mc
    error_true_to_critic[i] = true_cri - critic
    error_true_to_mc[i] = true_cri - mc
    if i % 5 == 0:
        print(i)

print(f"Average critic-to-Monte Carlo error: {(error_critic_to_mc**2).mean()}")
print(f"Average true-to-Monte Carlo error: {(error_true_to_mc**2).mean()}")
print(f"Average true-to-critic error: {(error_true_to_critic**2).mean()}")

plt.style.use('ggplot')

plt.hist(error_true_to_critic.detach().cpu().numpy(), bins=30)
plt.ylabel("Frequency")
plt.xlabel("True value minus critic")
plt.show()
plt.hist(error_true_to_mc.detach().cpu().numpy(), bins=30)
plt.ylabel("Frequency")
plt.xlabel("True value minus Monte Carlo estimate")
plt.show()
plt.hist(error_critic_to_mc.detach().cpu().numpy(), bins=30)
plt.ylabel("Frequency")
plt.xlabel("Critic minus Monte Carlo estimate")
plt.show()