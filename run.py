import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import math



# Util funcs


def input(gamma:float)->torch.Tensor:
  gamma=torch.Tensor([gamma])
  x=torch.tensor([torch.cos(gamma),torch.sin(gamma)])
  return x

def flatten(ls):
  return [item for sublist in ls for item in sublist]


# Simple Mean Loss function

def mean_loss(inp):
    return torch.mean(inp)

# Loss Wrapper function, returns the loss and the gradient of loss wrt net output

class LossWrapper(nn.Module):
    def __init__(self, loss_function):
      super(LossWrapper, self).__init__()
      self.loss_function = loss_function

    def loss_with_costgrad(self, net_out):
        L = net_out.shape[0]
        ret_loss = self.loss_function(net_out)

        nout = net_out.clone().detach()
        nout.requires_grad_(True)
        loss_out = self.loss_function(nout)
        nout.retain_grad()
        loss_out.backward()

        return ret_loss, nout.grad      
        

torch.use_deterministic_algorithms=True

def create_model(width, depth, seed, activation:nn.Module):
  torch.manual_seed(seed)
  random.seed(seed)
  # Append input layer with array of hidden layer and output layer
  mdl = nn.Sequential(
        *([nn.Linear(2, width), activation()] \
        + flatten([[nn.Linear(width, width), activation()] for k in range(depth - 2)]) \
        + [nn.Linear(width, 2)])
  )

  return mdl

def reset_grads(model):
   for name, param in model.named_parameters():
        if "weight" in name:
            param.grad = None

def get_cost_grads(model):
    dCost = []
    for name, param in model.named_parameters():
        if "weight" in name:
            dCost.append(param.grad.flatten())

    dCost = torch.concatenate(dCost)
    return dCost
    

def undo_first_in_chainrule(grad, dCdF, dev):
    dFunc = torch.stack([
        grad / torch.Tensor([dCdF[i]]).to(dev) for i in range(dCdF.shape[0])
    ])
    
    return dFunc.to(dev)
   
def get_NTK(model, loss_function, ref_input, input, device):
   
   lfn = LossWrapper(loss_function)

   out1 = model.forward(ref_input)
  #  lfn.loss_with_costgrad(out1)
   loss, g_loss = lfn.loss_with_costgrad(out1) # return loss and gradient of loss wrt function output

   loss.backward()
   cost_grads_1 = get_cost_grads(model) # dC/dtheta
   N1 = undo_first_in_chainrule(cost_grads_1, g_loss, device) # undoing first chainrule to get vector of dFunctionOutput/dTheta (actual NTK element)

   out2 = model.forward(input)
   loss2, g_loss2 = lfn.loss_with_costgrad(out2)
   
   loss2.backward()
   cost_grads_2 = get_cost_grads(model)
   N2 = undo_first_in_chainrule(cost_grads_2, g_loss2, device)

   # Made adjustment for 2d NTK
   NTK = torch.matmul(N1, torch.t(N2))
   reset_grads(model)
   return NTK


def doNTK(model, loss_function, device, gamma_spacing:float=0.01)->torch.Tensor:
  gammas = torch.arange(-1, 1, gamma_spacing)
  
  NTK_arr = [
     get_NTK(model, loss_function, input(0.0).to(device), input(gamma).to(device), device) for gamma in gammas
  ]
  
  return gammas, NTK_arr

def doNTK_surface(model, loss_function, device, numpts)->torch.Tensor:
  X = np.linspace(-1, 1, numpts, dtype=np.float32)
  Y = np.linspace(-1, 1, numpts, dtype=np.float32)
  gX, gY = np.meshgrid(X, Y)

  gZ = [
     [get_NTK(model, loss_function, torch.tensor([1.0, 0.0]).to(device), torch.tensor([_X, _Y]).to(device), device)[0, 0].item() for _X in X] for _Y in Y
  ]
  
  return gX, gY, np.array(gZ)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

########################################### ADJUST HERE
ACTIVATION = nn.ReLU
WIDTHS = [100, 500, 1000, 1500]
DEPTH = 4
SEED = 32
LOSS = mean_loss
SURFACE_POINTS = 50
###########################################

for W in WIDTHS:
    mod = create_model(W, DEPTH, SEED, ACTIVATION).to(device)
    X, Y, Z = doNTK_surface(mod, LOSS, device, SURFACE_POINTS)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    s = "img/width_"+str(W)+"_Act_"+ACTIVATION.__name__
    plt.savefig(s+"_surface.png")
    plt.clf()

    gap = 2.0 / SURFACE_POINTS
    X, Y = doNTK(mod, LOSS, device, gap)
    bY = [Y[n][0, 0].item() for n in range(len(Y))]
    plt.plot(X, bY)
    plt.savefig(s+".png")
    print("Saved with width "+str(W))
    plt.clf()

