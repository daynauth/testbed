import torch
import re
from torch import profiler
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

def extract_time(line):
    time = re.findall(r"[-+]?\d*\.\d+|\d+", line)

    assert (len(time) == 1), "Incorrect time formal"
    time = time[0]

    time_pos = line.find(str(time))
    time_unit = line[time_pos+len(str(time)):]

    if time_unit != "ms":
        if(time_unit == "s"):
            time = float(time) * 1000
        else:
            raise ValueError("Time unit not recognized")

        
    return time

def time_profile(model_name, batch_size):
    model = models.__dict__[model_name]().cuda()
    inputs = torch.rand(batch_size, 3, 300, 300).cuda()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("model_inference"):
            model(inputs)

    table = prof.key_averages().table(sort_by="cuda_time_total").splitlines()

    return extract_time(table[-2]), extract_time(table[-1])

cpu_time, cuda_time = time_profile('resnet50', 5)
print("cpu time: ", cpu_time, "ms")
print("cuda time: ", cuda_time, "ms")