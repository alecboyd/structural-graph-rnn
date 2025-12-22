import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

x = torch.randn(3, 3, device=device)
print("Random tensor:\n", x)
