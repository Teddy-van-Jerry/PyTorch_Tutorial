# PyTorch Tutorial
# Chapter 3: Autograd
# https://youtu.be/c36lUUr864M
# Reimplemented by Teddy van Jerry
# 2021-08-27

import torch

x = torch.randn(3, requires_grad = True)
print(x)

y = x + 2
print(y)
z = y * y * 2
print(z)
v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float32)
z.backward(v) # dz/dx
print(x.grad)
print('----------------------------')

# If we do not want it to have grad function:
# Way I
x.requires_grad_(False)
print(x)

x.requires_grad_(True) # reset

# Way II
y = z.detach()
print(y)

# Way III
with torch.no_grad():
    y = x + 2
    print(y) # no grad

# By comparision
y = x + 2
print(y) # with grad
print('----------------------------')

weights = torch.ones(3, requires_grad = True)

print('Wrong Example:')
for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    # We can see grad values are sumed up,
    # and that is not what we want
print('----------------------------')

print('Right Example:')
for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() # empty grad values
