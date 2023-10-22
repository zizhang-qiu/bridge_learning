import torch

label = torch.tensor([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0]])
pred = torch.tensor([[0.3, 0.2, 0.4, 0.1],
                     [0.4, 0.2, 0.3, 0.1],
                     [0.2, 0.5, 0.2, 0.1]])
print(torch.log(pred) * label)
loss = -torch.sum(torch.log(pred) * label, 1).mean()
print(loss)
