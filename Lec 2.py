# Pytorch Toturial
# Chapter 2: Tensor Basics
# https://youtu.be/c36lUUr864M
# Reimplemented by Teddy van Jerry
# 2021-08-26

import torch
import numpy as np

x = torch.rand(2, 2)
y = torch.rand(2, 2)
a = torch.tensor([2, 4.5])
print(x)
print(y)
print(a)
print('----------------------------')

z = x + y
y.add_(x) # in-place addition
z = torch.add(x, y) # same as the 'z = x + y'
print(z)
z = x - y
z = torch.sub(x, y)
print(z)
y.mul_(z)
z = torch.divide(x, y)
print('----------------------------')

x = torch.rand(5, 3)
print(x)
# Operations below are similar to MATLAB
print(x[:, 0])
print(x[1, :])
print(x[2, 2])
print(x[2, 2].item()) # get the value
print('----------------------------')

x = torch.rand(4, 4)
print(x)
y = x.view(16) # reshape into one dimension
print(y)
y = x.view(-1, 8) # the other dimension will be auto determined
print(y.size())
print('----------------------------')

a = torch.ones(5)
print(a)
b = a.numpy()
print(b, type(b)) # convert to numpy
# Note that if we are using CPU,
# they are on the same momory,
# so that is similar to reference in C++.
a.add_(1)
print(a, b) # b changed with a
a = np.ones(5)
b = torch.from_numpy(a)
a += 1
print(a, b) # a changed with b
if torch.cuda.is_available():
    mydevice = torch.device("cuda")
    x = torch.ones(5, dtype = torch.complex64, device = mydevice)
    y = torch.ones(5)
    y = y.to(mydevice)
    z = x + y
    # Error: numpy can only use CPU 
    # z.numpy()
    z = z.to("cpu")
    print(z, z.numpy())
print('----------------------------')

x = torch.ones(5, requires_grad = True)
print(x)
