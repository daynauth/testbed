import torch
import re
import pandas as pd
import matplotlib as plt
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
        if time_unit == "s":
            time = float(time) * 1000
        elif time_unit == "us":
            time = float(time) / 1000
        else:
            print(time_unit)
            raise ValueError("Time unit not recognized")

        
    return float(time)

def time_profile(model_name, batch_size):
    model = models.__dict__[model_name]().cuda()
    inputs = torch.rand(batch_size, 3, 300, 300).cuda()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("model_inference"):
            model(inputs)

    table = prof.key_averages().table(sort_by="cuda_time_total").splitlines()

    return extract_time(table[-2]), extract_time(table[-1])

model_list = [
    'resnet50',
    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small'
]

#initial test usually takes longer
time_profile("resnet50", 5)

batches = [1, 5, 10, 50]

data = []
for model_name in model_list:
    for batch_size in batches:
        cpu_time, cuda_time = time_profile(model_name, batch_size)
        data.append([model_name, batch_size, cpu_time, cuda_time])

#generate dataframe
df = pd.DataFrame(data, columns=['model', 'batch_size', 'cpu_time', 'cuda_time'])

output_dir = 'graphs'

for batch in batches:
    df1 = df[df['batch_size'] == batch]
    df1 = df1.drop(['batch_size'], axis=1)
    df1 = df1.set_index('model')

    title = 'Runtime Comparison at batch = ' + str(batch)
    ax = df1.plot.bar(stacked = True, rot=0, fontsize = 8, color = ['#f54242', '#3f5cfc'], title=title)

    ax.set_xlabel('Models')
    ax.set_ylabel('Runtime (ms)')
    ax.grid(visible=True, which='major', axis='y', linestyle = '-.', alpha = .25, zorder = 3)

    fig = ax.get_figure()
    
    fig.savefig(output_dir + '/' + 'runtime_stacked_batch_' + str(batch) + '.png')
