import torch

# Example 'Data' tensor of size B x A x A x T x T containing indices (e.g., in range [0, 87])
B, A, T = 2, 3, 4  # Example dimensions
tmp=torch.tensor([[80,80],[80,80],[80,80]]) # Random tensor of indices in the range [0, 87] of size 2x2

print(torch.randint(0,88,(64,34,34)))

agent_width_dict = torch.zeros(88)
agent_width_dict[[48, 66, 77, 80, 86]] = torch.tensor([0.0, 0.6, 0.8, 0.6, 1.7])

# Map tmp indices to corresponding values
thresh = agent_width_dict[tmp]
print(thresh)


#and

#thresh = agent_width_dict[T_agents.cpu()]

#where T_agents.cpu() is a tensor of shape 64x34x34x60x60 and dtype of torch.uint8