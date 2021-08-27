# PyTorch Tutorial
# Chapter 4: Backpropagation
# https://youtu.be/c36lUUr864M
# Reimplemented by Teddy van Jerry
# 2021-08-27

import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad = True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y) ** 2

print(loss)

# backward pass
loss.backward()
print(w.grad)
