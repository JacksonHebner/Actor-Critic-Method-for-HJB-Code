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

dim = 10 #dimension of state space/domain Omega
dim_ac = 1 #dimension of the action space A
dim_bm = 1 #dimension of the Brownian motion W_t driving the SDE
N_act = 2**9
N_cri = 2**9
beta_net = 0.6

# HJB equation parameters
gamma = 1
R = 1

# Initialization
#qnet = ACnet(dim, N_cri, 1, beta_net)
qnet = SimpleNN(dim_in = dim, num_neurons= 512, num_layers=2, dim_out = 1, bias = True)
qparams = dict(qnet.named_parameters())

unet = ACnet(dim, N_act, dim_ac, beta_net, bias = True)
#unet = SimpleNN(dim_in = dim, num_neurons= 50, num_layers=2, dim_out = dim_ac)
uparams = dict(unet.named_parameters())

#Hyperparameter functions and their vectorized counterparts
def eta_point(point):
    return R ** 2 - torch.sum(point ** 2)
def f(point): #extrapolated boundary condition
    return math.sin(R**2)
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

def true_critic(point):
    return torch.sin(torch.sum(point ** 2))
def true_actor(point):
    return torch.log(torch.sum(torch.exp(point)))

true_critic_vec = vmap(true_critic)
true_actor_vec = vmap(true_actor)

#HJB coefficients
def drift(point, action): #outputs dim-dimensional vector
    return action * torch.sin(point)
def diffusion(point, action): #outputs dim x dim_bm matrix
    output_vec = 1 + point**2
    return torch.reshape(output_vec, (-1,dim_bm))
def zeta(point, action): #outputs dim_ac-dimensional vector
    return torch.log(torch.cosh(action - true_actor(point)))

true_hess, true_grad = func.hessian(true_critic), func.grad(true_critic)
def cost(point, action):
    dif = diffusion(point, action)
    diff_term = torch.matmul(dif, dif.transpose(0,1)) * true_hess(point)
    return zeta(point, action) + gamma * true_critic(point) - drift(point, action).dot(true_grad(point)) - diff_term.sum() / 2

drift_vec, diffusion_vec, zeta_vec, cost_vec = vmap(drift), vmap(diffusion), vmap(zeta), vmap(cost)

#Generic PDE operator and Hamiltonian functions, in unvectorized and vectorized form
critic_hess, critic_grad = func.hessian(critic_point_func, argnums=2), func.grad(critic_point_func, argnums=2)
def pde(point, qnet, qparams, unet, uparams):
    action = actor_point_func(unet, uparams, point)

    dif = diffusion(point, action)
    diff_term = (torch.matmul(dif, dif.transpose(0,1)) * critic_hess(qnet, qparams, point)).sum() / 2

    dri = drift(point, action)
    drift_term = dri.dot(critic_grad(qnet, qparams, point))

    cos = cost(point, action)
    poi = critic_point_func(qnet, qparams, point)

    return diff_term + drift_term + cos - gamma * poi

pde_vec = vmap(pde, in_dims = (0, None, None, None, None))

def ham(point, qnet, qparams, unet, uparams):
    action = actor_point_func(unet, uparams, point)

    dif = diffusion(point, action)
    diff_term = (torch.matmul(dif, dif.transpose(0,1)) * critic_hess(qnet, qparams, point).detach()).sum() / 2

    dri = drift(point, action)
    drift_term = dri.dot(critic_grad(qnet, qparams, point).detach())

    cos = cost(point, action)

    return diff_term + drift_term + cos

ham_vec = vmap(ham, in_dims = (0, None, None, None, None))

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

DGM = False

#Optimzers and schedulers
Qoptimizer = optim.Adam(qnet.parameters(), lr=initial_lr_critic)
Qscheduler = LambdaLR(Qoptimizer, lr_lambda=lambda epoch: initial_lr_critic * (epoch // Num_epo_cri + 10) ** (-.8))
Uoptimizer = optim.Adam(unet.parameters(), lr=initial_lr_actor)
Uscheduler = LambdaLR(Uoptimizer, lr_lambda=lambda epoch: initial_lr_actor * (epoch // Num_epo_act + 10) ** (-.6))

for j in range(start, start + Num_iterate):
    source = R * sample_unit_ball(dim, Nbasepoints)

    # Critic step:
    if (j%2):
        # Critic training
        for count1 in tqdm(range(Num_epo_cri)):
            grid = R * sample_unit_ball(dim, Nmc)
            grid.requires_grad = True

            #Slight redundancy -- could be avoided by rewriting pde_vec, but I wanted to keep pde_vec and ham_vec similar
            out = critic_vec(qnet, qparams, grid)

            # Evaluate PDE operator
            op = pde_vec(grid, qnet, qparams, unet, uparams)

            #True Q-PDE uses a smooth clipping function, but that is hard to implement in code
            if DGM:
                loss_to_min = torch.dot(op, op) / Nmc
                loss_DGM = loss_to_min
            else:
                lq = op.detach()
                loss_to_min = torch.dot(-lq, out) / Nmc
                loss_DGM = torch.dot(lq,lq) / Nmc

            # Critic update
            Qoptimizer.zero_grad()
            loss_to_min.backward()
            Qoptimizer.step()
            Qscheduler.step()

            # memory optimization (more to do)
            loss_to_min = loss_to_min.detach()
            op = op.detach()
            out = out.detach()

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
            big_critic_op_loss.append(float(loss_DGM))

    # Actor step:
    if not (j%2):
        # Actor training
        for count2 in tqdm(range(Num_epo_act)):
            grid = R * sample_unit_ball(dim, Nmc)
            grid.requires_grad = True

            # Evaluate integral-Hamiltonian
            ha = torch.clamp(ham_vec(grid, qnet, qparams, unet, uparams), min = -100)
            ih = ha.mean()
            #ih = torch.mean(ham_vec(grid, qnet, qparams, unet, uparams))

            # Critic update
            Uoptimizer.zero_grad()
            ih.backward()
            Uoptimizer.step()
            Uscheduler.step()

            # memory optimization (more to do)
            ih = ih.detach()

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

plt.style.use('ggplot')

plt.figure(figsize=(12,8))
plt.ylabel('Critic errors')
plt.xlabel('Critic update epoch')
plt.yscale('log')
plt.plot(big_critic_loss, label = "Critic mean square error")
plt.plot(big_critic_op_loss, label = "Critic loss function")
plt.legend(loc = "upper left")
plt.show()

plt.figure(figsize=(12,8))
plt.ylabel('Actor errors')
plt.xlabel('Actor update epoch')
plt.yscale('log')
plt.plot(big_actor_loss, label = "Actor mean square error")
plt.plot(big_actor_ham_loss, label = "Actor loss function")
plt.legend(loc = "upper left")
plt.show()

x = torch.linspace(-R, R, 1000).unsqueeze(1)
zeros = torch.zeros(1000, dim - 1)
test_tensor = torch.cat((x, zeros), dim=1)

test_critic = critic_vec(qnet, qparams, test_tensor).detach().cpu().numpy()
test_critic_true = true_critic_vec(test_tensor).detach().cpu().numpy()
test_actor = actor_vec(unet, uparams, test_tensor).detach().cpu().numpy()
test_actor_true = true_actor_vec(test_tensor).detach().cpu().numpy()
x = x.detach().cpu().numpy().squeeze()

plt.figure(figsize=(10,7))
plt.ylabel('Critic (value) function')
plt.xlabel('First coordinate value (all others zero)')
plt.yscale('linear')
plt.plot(x, test_critic, label = 'Critic estimate')
plt.plot(x, test_critic_true, label = 'True value function')
plt.legend(loc = "upper left")
plt.show()

plt.figure(figsize=(10,7))
plt.ylabel('Actor (control) function')
plt.xlabel('First coordinate value (all others zero)')
plt.yscale('linear')
plt.plot(x, test_actor, label = 'Actor estimate')
plt.plot(x, test_actor_true, label = 'True optimal control')
plt.legend(loc = "upper left")
plt.show()