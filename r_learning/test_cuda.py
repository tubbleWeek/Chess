import torch
DEVICE = torch.device("cuda")
cuda_id = torch.cuda.current_device()
print(torch.cuda.is_available())
print(DEVICE)
print(f'Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}')