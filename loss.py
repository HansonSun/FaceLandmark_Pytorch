import torch
l1 = torch.FloatTensor(((0.1,0.1),(0.2,0.2),(0.3,0.3),(0.4,0.4),(0.5,0.5)))
l2 = torch.FloatTensor(((0.18,0.18),(0.28,0.28),(0.38,0.38),(0.48,0.48),(0.58,0.58)))
l3 = torch.FloatTensor(((0.11,0.12),(0.21,0.22),(0.31,0.32),(0.41,0.42),(0.51,0.52)))
l4 = torch.FloatTensor(((0.102,0.102),(0.202,0.202),(0.302,0.302),(0.402,0.402),(0.502,0.502)))
criterion = torch.nn.MSELoss()


print(criterion(l1,l2))
print(criterion(l1,l3))
print(criterion(l1,l4))