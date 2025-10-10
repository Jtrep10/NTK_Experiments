import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import math

# Simple Mean Loss function

class MeanLoss(nn.Module):
    def __init__(self):
        super(MeanLoss, self).__init__()
        
    def forward(self, input_tensor):
        loss_out = torch.mean(input_tensor)
        return loss_out
    
    def loss_with_costgrad(self, net_out):
        L = net_out.shape[0]
        loss_out = self.forward(net_out)
        dloss_out = torch.tensor([
           1.0 / L for _ in range(L)
        ])
        return loss_out, dloss_out
    

torch.use_deterministic_algorithms=True

def input(gamma:float)->torch.Tensor:
  gamma=torch.Tensor([gamma])
  x=torch.tensor([torch.cos(gamma),torch.sin(gamma)])
  return x

def flatten(ls):
  return [item for sublist in ls for item in sublist]

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
   
def get_NTK(model, ref_input, input, device):
   loss_function = MeanLoss()

   out1 = model.forward(ref_input)
   loss, g_loss = loss_function.loss_with_costgrad(out1) # return loss and gradient of loss wrt function output

   loss.backward()
   cost_grads_1 = get_cost_grads(model) # dC/dtheta
   N1 = undo_first_in_chainrule(cost_grads_1, g_loss, device) # undoing first chainrule to get vector of dFunctionOutput/dTheta (actual NTK element)

   out2 = model.forward(input)
   loss2, g_loss2 = loss_function.loss_with_costgrad(out2)
   
   loss2.backward()
   cost_grads_2 = get_cost_grads(model)
   N2 = undo_first_in_chainrule(cost_grads_2, g_loss2, device)

   # Made adjustment for 2d NTK
   NTK = torch.matmul(N1, torch.t(N2))
   reset_grads(model)
   return NTK


def doNTK(model, device, gamma_spacing:float=0.01)->torch.Tensor:
  gammas = torch.arange(-1, 1, gamma_spacing)
  
  NTK_arr = [
     get_NTK(model, input(0.0).to(device), input(gamma).to(device), device) for gamma in gammas
  ]
  
  return gammas, NTK_arr

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

mod = create_model(100, 4, 32, nn.ReLU).to(device)
X, Y = doNTK(mod, device, 0.01)
bY = [Y[n][0, 0].item() for n in range(len(Y))]

plt.plot(X,bY)
plt.savefig("img/width100.png")
print("Saved with width 100.")
plt.clf()

mod = create_model(500, 4, 32, nn.ReLU).to(device)
X, Y = doNTK(mod, device, 0.01)
bY = [Y[n][0, 0].item() for n in range(len(Y))]

plt.plot(X,bY)
plt.savefig("img/width500.png")
print("Saved with width 500.")
plt.clf()


mod = create_model(1000, 4, 32, nn.ReLU).to(device)
X, Y = doNTK(mod, device, 0.01)
bY = [Y[n][0, 0].item() for n in range(len(Y))]

plt.plot(X,bY)
plt.savefig("img/width1000.png")
print("Saved with width 1000.")
plt.clf()


mod = create_model(1500, 4, 32, nn.ReLU).to(device)
X, Y = doNTK(mod, device, 0.01)
bY = [Y[n][0, 0].item() for n in range(len(Y))]

plt.plot(X,bY)
plt.savefig("img/width1500.png")
print("Saved with width 1500.")
plt.clf()


mod = create_model(2000, 4, 32, nn.ReLU).to(device)
X, Y = doNTK(mod, device, 0.01)
bY = [Y[n][0, 0].item() for n in range(len(Y))]

plt.plot(X,bY)
plt.savefig("img/width2000.png")
print("Saved with width 2000.")
plt.clf()


# res = mod.forward(input(gamma))

# n = res.shape[0]

# loss = loss_func(res)

# loss.backward()

# # now we calculate array of dC/dtheta

# dCost = []
# for name, param in mod.named_parameters():
#    if "weight" in name:
#       dCost.append(param.grad.flatten())

# dCost = torch.concatenate(dCost)

# print(dCost)

# # then because dC/dtheta is a result of chain rule, we can finally divide dC/dtheta with dC/dOutput to get dOutput/dtheta, which is the NTK
# # dMean is [1 / n, 1 / n, 1 / n...]

# dMean = torch.Tensor([1.0 / n for k in range(n)])

# dFunc = torch.stack([
#    dCost / torch.Tensor([dMean[i]]) for i in range(dMean.shape[0])
# ])
