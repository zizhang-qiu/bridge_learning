"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: temp2.py
@time: 2024/1/17 15:06
"""
import torch
import torch.nn.functional as F

# 定义行数和每行的元素个数
random_targets = torch.ones(size=[5, 52])

indices = torch.multinomial(random_targets, num_samples=13, replacement=False)
print(indices)

random_targets.scatter_(1, indices, 0)
random_targets = 1-random_targets

input = torch.randn(5, 52, requires_grad=True)
input = torch.sigmoid(input)
# target = torch.randn(3, 5).softmax(dim=1)
print(input, random_targets, sep="\n")
loss = F.cross_entropy(input, random_targets, reduction="mean")
print(loss)
loss2 = F.binary_cross_entropy(input, random_targets)
print(loss2)

target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
