"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: temp2.py
@time: 2024/1/17 15:06
"""
import torch



probs = torch.rand(size=(4, 3))
probs = torch.nn.functional.softmax(probs, dim=-1)

print(probs)