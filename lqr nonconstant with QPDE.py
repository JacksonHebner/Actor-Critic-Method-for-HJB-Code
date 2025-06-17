import numpy as np
import torch
from torch.func import vmap, grad, functional_call
import torch.func as func
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm, trange
from math import exp, sqrt, log
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from neural_network_classes import ACnet
from neural_network_classes import SimpleNN

torch.set_default_device("cpu")

dim = 10
N_act = 2**9
N_cri = 2**9
Num_epo_cri = 100
Num_epo_act = 200
initial_lr = 0.1
beta_net = 0.6

# LQR equation parameters
gamma = 1
beta = 1
p = 1
q = 1
R = 1
k = (sqrt(q**2 * gamma**2 + 4*p*q*beta**2) - gamma * q)/(2*beta**2)
epsilon = 0
#functions to later use in evaluating PDE loss, and their vectorized equivalents
def f_tilde(x):
    return gamma * k * torch.sum(x**2) + torch.sum((k**2 * (beta + 2 * epsilon)**2 * x**2) / (q + 2*k*epsilon**2 * x**2)) - 2*k*dim
f_tilde_vec = vmap(f_tilde, in_dims = 0)
def second_deriv_sum(hessian_diag, control, point):
    return torch.dot(hessian_diag, (1 + epsilon * control * point)**2)
second_deriv_sum_vec = vmap(second_deriv_sum, in_dims = (0, 0, 0))

# Monte Carlo
Nbasepoints = int(1e6)
Nmc = 3000
max_grad = 1000000

def sample_unit_ball(n_dims, n_samples): #samples uniformly
    gaussian_points = torch.randn(n_samples, n_dims)
    sphere_points = gaussian_points / torch.linalg.norm(gaussian_points, dim=1, keepdim=True)
    radii = torch.rand(n_samples, 1) ** (1/n_dims)

    return sphere_points * radii

def eta_point(point):
    return 1 - torch.sum(point ** 2)

def f(point):
    return k*R**2

# Initialization
qnet = ACnet(dim, N_cri, 1, beta_net)
#qnet = SimpleNN(dim_in = dim, num_neurons= N_cri, num_layers=2, dim_out = 1)
qparams = dict(qnet.named_parameters())

unet = ACnet(dim, N_act, dim, beta_net)
#unet = SimpleNN(dim_in = dim, num_neurons= N_act, num_layers=2, dim_out = dim)
uparams = dict(unet.named_parameters())

def critic_point_func(qnet, qparams, point):
    qnet_output = functional_call(qnet, qparams, (point.unsqueeze(0),))
    scalar_out = qnet_output.squeeze()  # shape []
    eta = eta_point(point)
    return eta * scalar_out + f(point)

def actor_point_func(unet, uparams, point):
    unet_output = functional_call(unet, uparams, (point.unsqueeze(0),))
    return unet_output.squeeze()  # shape []

critic_vec = vmap(critic_point_func, in_dims = (None, None, 0))
actor_vec = vmap(actor_point_func, in_dims = (None, None, 0))

#vectorized, functional calls
grad_func = vmap(func.grad(critic_point_func, argnums=2), in_dims = (None, None, 0))
jacob_func = vmap(func.jacrev(critic_point_func, argnums=2), in_dims = (None, None, 0))
hessian_func = vmap(func.hessian(critic_point_func, argnums=2), in_dims = (None, None, 0)) #returns a vector of matrices

batch_diag = vmap(torch.diagonal)

def bad_hess_diag_func(qnet, qparams, grid): #inefficient hessian diagonal function
    hess = hessian_func(qnet, qparams, grid)
    return batch_diag(hess)

def true_critic(point):
    return k * torch.sum(point ** 2)

def true_actor(point):
    return -(beta + 2*epsilon) * point / (q/k + 2*epsilon**2 * point ** 2)

true_critic_vec = vmap(true_critic)
true_actor_vec = vmap(true_actor)

start = 0 # This controls order of training a and c: 0 -> actor first, 1 -> critic first
Num_iterate = 100
big_critic_loss, big_actor_loss = [], []
big_rel_critic_loss, big_rel_actor_loss = [], []

#Optimzers and schedulers
Qoptimizer = optim.Adam(qnet.parameters(), lr=initial_lr)
Qscheduler = LambdaLR(Qoptimizer, lr_lambda=lambda epoch: initial_lr * (epoch // Num_epo_cri + 1) ** (-1.0))
Uoptimizer = optim.Adam(unet.parameters(), lr=initial_lr)
Uscheduler = LambdaLR(Uoptimizer, lr_lambda=lambda epoch: initial_lr * (epoch // Num_epo_act + 1) ** (-1.0))

for j in range(start, start + Num_iterate):
    source = sample_unit_ball(dim, Nbasepoints)

    # Critic step:
    if (j%2):
        # Critic training
        for count1 in tqdm(range(Num_epo_cri)):
            grid = sample_unit_ball(dim, Nmc)
            grid.requires_grad = True

            # Net output
            out = critic_vec(qnet, qparams, grid)
            uout = actor_vec(unet, uparams, grid)

            #calculate derivatives
            grad_matrix = grad_func(qnet, qparams, grid)
            second_order = second_deriv_sum_vec(bad_hess_diag_func(qnet, qparams, grid), uout, grid)
            u_norms = torch.sum(uout**2, dim = 1).detach()
            f_tilde_vals = f_tilde_vec(grid)

            # Evaluate PDE operator
            op = second_order + beta*torch.sum(grad_matrix * uout, dim = 1) + q*u_norms + f_tilde_vals - gamma*out
            lq = op.detach()
            loss_to_min = torch.mean(-lq * out)

            # Critic update
            Qoptimizer.zero_grad()
            loss_to_min.backward()
            Qoptimizer.step()
            Qscheduler.step()

            # memory optimization
            loss_to_min = loss_to_min.detach()
            op = op.detach()
            second_order = second_order.detach()
            f_tilde_vals = f_tilde_vals.detach()

            #lazy functional programming, but it works
            qparams = dict(qnet.named_parameters())

        # Comparison against true value function
        with torch.no_grad():
            critic_estimate = critic_vec(qnet, qparams, source)
            critic_true = true_critic_vec(source)
            critic_loss = torch.mean((critic_estimate - critic_true)**2).detach().cpu().numpy()
            rel_critic_loss = torch.sum((critic_estimate - critic_true)**2)/ torch.sum(critic_true**2)
            big_critic_loss.append(float(critic_loss))
            big_rel_critic_loss.append(float(rel_critic_loss.detach().cpu().numpy())) ## Easier to graph floats than tensors

    # Actor step:
    if not (j%2):
        # Actor training
        for count2 in tqdm(range(Num_epo_act)):
            grid = sample_unit_ball(dim, Nmc)
            grid.requires_grad = True

            # Net output
            out = critic_vec(qnet, qparams, grid)
            uout = actor_vec(unet, uparams, grid)

            # calculate derivatives
            grad_matrix = grad_func(qnet, qparams, grid).detach()
            diag = bad_hess_diag_func(qnet, qparams, grid).detach()
            second_order = second_deriv_sum_vec(diag, uout, grid)
            u_norms = torch.sum(uout ** 2, dim=1)
            x_norms = torch.sum(grid ** 2, dim=1)

            # Evaluate integral-Hamiltonian
            ih = torch.mean(second_order + beta * torch.sum(grad_matrix * uout, dim=1) + q * u_norms)

            # Critic update
            Uoptimizer.zero_grad()
            ih.backward()
            Uoptimizer.step()
            Uscheduler.step()

            # memory optimization
            ih = ih.detach()
            second_order = second_order.detach()

            # lazy functional programming, but it works
            uparams = dict(unet.named_parameters())

        # Comparison against true control
        with torch.no_grad():
            actor_estimate = actor_vec(unet, uparams, source)
            actor_true = true_actor_vec(source)
            actor_loss = torch.mean(torch.sum((actor_estimate - actor_true) ** 2, dim = 1)).detach().cpu().numpy()
            rel_actor_loss = torch.sum((actor_estimate - actor_true) ** 2) / torch.sum(actor_true ** 2)
            big_actor_loss.append(float(actor_loss))
            big_rel_actor_loss.append(float(rel_actor_loss.detach().cpu().numpy()))

    if ((j % 5 == 0) & (j > 2)):
        print(f"Iteration {j + 1 - start}")
        print(f"Critic mean square error: {critic_loss}")
        print(f"Critic relative error: {rel_critic_loss}")
        print(f"Actor mean square error: {actor_loss}")
        print(f"Actor relative error: {rel_actor_loss}")

x = torch.linspace(-R, R, 1000).unsqueeze(1)
zeros = torch.zeros(1000, dim - 1)
test_tensor = torch.cat((x, zeros), dim=1)

test_critic = critic_vec(qnet, qparams, test_tensor).detach().cpu().numpy()
test_critic_true = true_critic_vec(test_tensor).detach().cpu().numpy()
test_actor = actor_vec(unet, uparams, test_tensor).detach().cpu().numpy()
test_actor_true = true_actor_vec(test_tensor).detach().cpu().numpy()
x = x.detach().cpu().numpy().squeeze()



plt.style.use('ggplot')

plt.figure(figsize=(10,7))
plt.ylabel('Critic (value) function')
plt.xlabel('First coordinate value (all others zero)')
plt.yscale('log')
plt.plot(x, test_critic, label = 'Critic estimate')
plt.plot(x, test_critic_true, label = 'True value function')
plt.legend()
plt.show()

plt.figure(figsize=(10,7))
plt.ylabel('Critic (value) function')
plt.xlabel('First coordinate value (all others zero)')
plt.yscale('linear')
plt.plot(x, test_critic, label = 'Critic estimate')
plt.plot(x, test_critic_true, label = 'True value function')
plt.legend()
plt.show()

plt.figure(figsize=(10,7))
plt.ylabel('First coordinate of actor (control) function')
plt.xlabel('First coordinate value (all others zero)')
plt.yscale('log')
plt.plot(x, test_actor[:,0], label = 'Actor estimate')
plt.plot(x, test_actor_true[:,0], label = 'True optimal control')
plt.legend()
plt.show()

plt.figure(figsize=(10,7))
plt.ylabel('First coordinate of actor (control) function')
plt.xlabel('First coordinate value (all others zero)')
plt.yscale('linear')
plt.plot(x, test_actor[:,0], label = 'Actor estimate')
plt.plot(x, test_actor_true[:,0], label = 'True optimal control')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.ylabel('Absolute critic estimate error')
plt.xlabel('First coordinate value (all others zero)')
plt.yscale('linear')
plt.plot(x, np.absolute(test_critic - test_critic_true), label = 'Absolute critic estimate error')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.ylabel('Absolute actor estimate error (in L^2)')
plt.xlabel('First coordinate value (all others zero)')
plt.yscale('linear')
plt.plot(x, np.sqrt(np.sum((test_actor - test_actor_true)**2, axis = 1)), label = 'Absolute actor estimate error')
plt.legend()
plt.show()




plt.figure(figsize=(12,8))
plt.ylabel('Critic mean square error')
plt.xlabel('Critic update epoch')
plt.yscale('log')
plt.plot(big_critic_loss)
plt.show()

plt.figure(figsize=(12,8))
plt.ylabel('Actor mean square error')
plt.xlabel('Actor update epoch')
plt.yscale('log')
plt.plot(big_actor_loss)
plt.show()