import torch

import time
class TimingUtil():
    def __init__(self):
        self.event_list=[]
        self.events={}

        self.t = None
        self.counter=None
        self.cycles=0

    def start(self):
        self.t = time.time()
        self.counter=0
        self.cycles += 1

    def tick(self, event_name=None):
        t = time.time()
        if event_name is None:
            event_name = str(self.counter)
        
        if event_name not in self.events:
            self.events[event_name] = {
                'name': event_name,
                'idx': self.counter,
                't': []
            }
        
        self.events[event_name]['t'].append(t - self.t)
        self.t = t
        self.counter += 1

    def report_times(self):
        for k, v in self.events.items():
            t_avg = sum(v['t']) / len(v['t'])
            self.events[k]['t_avg'] = t_avg
        
        print("Average Times over %d cycles:" % self.cycles)
        for k, v in self.events.items():
            print("\t%s: %f seconds." % (v['name'], v['t_avg']))






def masked_softmax(x, mask, dim=-1, eps=1e-8):
    x_masked = x.clone()
    x_masked = x_masked - x_masked.max(dim=dim, keepdim=True)[0]
    x_masked[mask == 0] = -float("inf")
    return torch.exp(x_masked) / (torch.exp(x_masked).sum(dim=dim, keepdim=True) + eps)


def to_images(inputs):
    return inputs.view(-1, *inputs.size()[-3:])

def to_set(inputs, set_dims=None, initial_set=None):
    if initial_set is not None:
        set_dims = initial_set.size()[:-3]
    return inputs.view(*set_dims, -1)

def to_imgset(inputs, set_dims=None, initial_set=None):
    if initial_set is not None:
        set_dims = initial_set.size()[:-3]
    return inputs.view(*set_dims, *inputs.size()[-3:])



def split_dataset(dataset, weights):
    assert sum(weights) == 1
    N = len(dataset)
    indices = torch.randperm(N)
    splits=[]
    tot=0
    for i in range(len(weights)):
        if i != len(weights) - 1:
            N_i = int(N * weights[i])
            indices_i = indices[tot:tot+N_i]
        else:
            indices_i = indices[tot:]
        splits.append(Subset(dataset, indices_i))
        tot += N_i
    return splits



def greatest_power_2(n):
    k = int(math.log(n, 2))
    for i in range(k):
        if n % 2**(i+1) != 0:
            return 2**i
    return 2**k



def load_state_partial(model, state_dict, exclude_keys=[]):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    valid_keys = [k for k,v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()]
    if len(exclude_keys) > 0:
        valid_keys = [k for k in valid_keys if not any([k.startswith(k_ex) for k_ex in exclude_keys])]
    state_dict = {k: v for k, v in state_dict.items() if k in valid_keys}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def submodule_state(submodule_name, state_dict):
    submodule_dict = {}
    for k, v in state_dict.items():
        new_key = ".".join(k.split(".")[1:])
        if new_key[:len(submodule_name)] == submodule_name:
            submodule_dict[new_key] = v
    return submodule_dict


import glob
import os
import re
def load_latest_ckpt(checkpoint_path, model_dir):
    base_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    step = base_checkpoint['step']
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint_*.pt"))
    checkpoint=None
    for checkpoint_i in checkpoints:
        ckpt_step = re.search("checkpoint_([a-z0-9]+).pt", checkpoint_i).group(1)
        if ckpt_step == "final":
            checkpoint=checkpoint_i
            break
        ckpt_step = int(ckpt_step)
        if ckpt_step > step:
            checkpoint=checkpoint_i
    checkpoint = base_checkpoint if checkpoint is None else torch.load(checkpoint, map_location='cpu')
    return checkpoint
