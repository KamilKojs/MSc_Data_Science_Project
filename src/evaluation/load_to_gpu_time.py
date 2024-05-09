import time
import torch
import torchvision.models as models

model = models.resnet152()
model.load_state_dict(torch.load(f'resnet152_weights_1.pth'))
model.eval()

start_time = time.time()
model.to("cuda:0")
end_time = time.time()
print(f"time to cuda: {end_time-start_time}")
print(torch.cuda.memory_allocated())

start_time = time.time()
del model
end_time = time.time()
print(f"time deletion: {end_time-start_time}")
print(torch.cuda.memory_allocated())