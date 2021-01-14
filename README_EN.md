# Distribution is all you need

## Take-Away

Note: The code mainly comes from：**[Here](https://github.com/tczhangzhi/pytorch-distributed)**, I fixed some bugs here and changed the dataset from ImageNet to CIFAR10 which can be easily downloaded:
1. fix the bug of apex when using ```class data_prefetcher```，remove the usage of ```data_prefetcher```，and changed to a easilier enumerate loop of using ```train_loader``` or ```val_loader```；
2. fix a bug of using horovod's all-reduce，since horovod.pytorch's ```allreduce``` method already includes average，it is not necessary to devidied by ```nprocs``` again.
3. added bash files，to easily run the py programs.

Tested under PyTorch, CIFAR10, NVIDIA DGX-1 with two configurations, 8 cards of 16GB V100, and 4cards of 16GB V100:

1. **[nn.DataParallel ](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/1.dataparallel.py) simple nn.DataParallel**
2. **[torch.distributed](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/2.distributed.py) using torch.distributed to speedup parallel training/inferencing**
3. **[torch.multiprocessing](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/3.multiprocessing_distributed.py) use torch.multiprocessing to replace the launcher from command line**
4. **[apex](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/4.apex_distributed2.py) use apex （fp16）to speedup further**
5. **[horovod](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/5.horovod_distributed.py)** **horovod implementation**
6. **[slurm](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/6.distributed_slurm_main.py) GPU cluster（Not Tested Yet! 2021/Jan/13）**
7. **append：distributed [evaluation](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/2.distributed.py)**

## 1. simple torch.nn.DataParallel

> DataParallel can help us（single-process control）pushing model and data into multiple GPUs，controlling data movements among GPUs，and cooperate GPUs' parallel training.

DataParallel is easy to use，we just need to include our model by DataParallel，and then set some parameters. The parameters include: available GPUs, device_ids=gpus; which GPU is used for reducing gradients，output_device=gpus[0]. DataParallel will help us cutting data into pieces and then loading to respective GPU，deliver model to GPU，forward and backward computing:

```
model = nn.DataParallel(model.cuda(), 
   device_ids=gpus, output_device=gpus[0])
```

Note that modle and data are requried to be loaded into GPU first，and then ```DataParallel``` 's module can process them (model + data):

```
# need model.cuda()
model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      # need images/target.cuda()
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

To conclude, DataParallel's parallel training requires the following code:

```
# 1.dataparallel.py
import torch
import torch.distributed as dist

gpus = [0, 1, 2, 3] # TODO need to be updated based on real-world gpu numbers
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

train_dataset = ...

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=...)

model = ...
model = nn.DataParallel(model.to(device), 
   device_ids=gpus, output_device=gpus[0])

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

When using, use python to execute：

```
python 1.dataparallel.py
```

Complete code using CIFAR10, click [Github](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/1.dataparallel.py)。

## 2. use torch.distributed for parallel training

> after pytorch 1.0, the official version supports: all-reduce，broadcast，send, and receive. MPI is used for CPU communication and NCCL for GPU communication. 

Different with ```DataParallel``` 's single-process controlling multiple GPUs, under ```distributed```, we just need to write one code and torch will help us arranging it to multi-GPUs.

At API level, pytorch supplies us ```torch.distributed.launch``` as the launcher，to run python code from command lines (linux). During running, the launcher will send the current process's index (aka gpu rank/index, ```local_rank```) to python, so that we can obtain the index of current process:

```
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
print(args.local_rank)
```

Then, we use ```init_process_group``` to set the backend and port of GPUs' communications:

```
dist.init_process_group(backend='nccl')
```

After that, ```DistributedSampler``` is used to split the dataset and perform distributed sampling. It will split a batch into several partitions. At current process, we only need to obtain the local_rank and the corresponding partition for training: 

```
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
```

Then, we can use ```DistributedDataParallel``` to update the model which can perform all-reduce among different GPUs' gradients. After all reduce, the gradients of GPUs are the average value of the gradients of GPUs before all-reduce：

```
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
```

Finally, push the model and data into current GPU, for forward and backward computing:

```
torch.cuda.set_device(args.local_rank)

model.cuda()

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

To conclude, ```torch.distributed``` parallel training code:

```
# 2.distributed.py
import torch
import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

model = ...
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

Call ```torch.distributed.launch``` to start-up the code (from command line):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 2.distributed.py
```

For full-version code under CIFAR10, click [Github](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/2.distributed.py)。

## 3. Use torch.multiprocessing to replace ```torch.distributed.launch```

We just need to call ```torch.multiprocessing.spawn```，```torch.multiprocessing``` will then help us automatically create processes, alike the following code. Below, ```spawn``` started nprocs=4 processes, each process will execute ```main_worker``` and took ```local_rank``` (current process's index = GPU's local rank) and args (i.e., 4 and myargs) as parameters:

```
import torch.multiprocessing as mp

mp.spawn(main_worker, nprocs=4, args=(4, myargs))
```

Here, the content that was managed by ```torch.distributed.launch``` are now at ```main_worker``` function, where proc = local_rank, and number-of-process nproc = 4， args = myargs:

```
def main_worker(proc, nproc, args):

   dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=4, rank=gpu)
   torch.cuda.set_device(args.local_rank)

   train_dataset = ...
   train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

   model = ...
   model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

   optimizer = optim.SGD(model.parameters())

   for epoch in range(100):
      for batch_idx, (data, target) in enumerate(train_loader):
          images = images.cuda(non_blocking=True)
          target = target.cuda(non_blocking=True)
          ...
          output = model(images)
          loss = criterion(output, target)
          ...
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
```

Note that there is no default environment values that ```torch.distributed.launch``` helped us reading, we need to manually set parameter values for ```init_process_group```:

```
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=4, rank=gpu)
```

To conclude, after importing ```multiprocessing```, the related parallel training code are:

```
# 3.multiprocessing_distributed.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

mp.spawn(main_worker, nprocs=4, args=(4, myargs))

def main_worker(proc, nprocs, args):

   dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', world_size=4, rank=gpu)
   torch.cuda.set_device(args.local_rank)

   train_dataset = ...
   train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

   model = ...
   model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

   optimizer = optim.SGD(model.parameters())

   for epoch in range(100):
      for batch_idx, (data, target) in enumerate(train_loader):
          images = images.cuda(non_blocking=True)
          target = target.cuda(non_blocking=True)
          ...
          output = model(images)
          loss = criterion(output, target)
          ...
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
```

For using, a simple python is enough:

```
python 3.multiprocessing_distributed.py.py
```

Complete code using CIFAR10, click [Github](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/3.multiprocessing_distributed.py)。

## Using Apex to further speed-up

> Apex is NVIDIA's open-source mixed-precision (fp32 and fp16) and distributed training lib. Apex is easy to be used with several lines' changing is enough. Also, Apex inlcudes code for distributed training, optimized for NVIDIA's NCCL communication protocol. 

Still it is not that easy to install [Apex](https://github.com/NVIDIA/apex), compared with horovod.

I faced bugs even during my installing of Apex:
```
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
NVCC is actually used for re-compiling. The bug that I faced was: nvcc's cuda version is different with pytorch's cuda! (cost time for finding this bug):

check nvcc's cuda version:
```
nvcc --v
ersion
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```

check pytorch's cuda:
```
python
Python 3.6.12 |Anaconda, Inc.| (default, Sep  8 2020, 23:10:56)
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.version.cuda
'10.1'
```
Since both are 10.1, I finally finished installing apex0.1 (2021/Jan/12). Before, I used pytorch with cuda11 and failed...

Another thing is that cuda is separated into two apis, one is running time support compile,  version 10.1.243. Another is that if use ```nvidia-smi``` you may see version 11.0>=10.1 (CUDA11)，the second CUDA11 is for driving GPU. It will be fine if we only ensure that nvidia-smi's cuda version >= nvcc's version.

当nvcc和pytorch的cuda版本不一致的时候，一则可以新安装cuda并修改nvcc的版本，二则可以在[pytorch安装界面](https://pytorch.org/get-started/locally/)，寻找和nvcc的版本一致的安装命令，reinstall pytorch(更简单一些）。



在混合精度训练上，Apex 的封装十分优雅。直接使用 amp.initialize 包装模型和优化器，apex 就会自动帮助我们管理模型参数和优化器的精度了，根据精度需求不同可以传入其他配置参数。

```
from apex import amp

model, optimizer = amp.initialize(model, optimizer)
```

在分布式训练的封装上，Apex 在胶水层的改动并不大，主要是优化了 NCCL 的通信。因此，大部分代码仍与 torch.distributed 保持一致。使用的时候只需要将 torch.nn.parallel.DistributedDataParallel 替换为 apex.parallel.DistributedDataParallel 用于包装模型。在 API 层面，相对于 torch.distributed ，它可以自动管理一些参数（可以少传一点）：

```
from apex.parallel import DistributedDataParallel

model = DistributedDataParallel(model)
# # torch.distributed
# model = torch.nn.parallel.DistributedDataParallel(model, 
#     device_ids=[args.local_rank])
# model = torch.nn.parallel.DistributedDataParallel(model, 
#     device_ids=[args.local_rank], output_device=args.local_rank)
```

During forward's loss computing, Apex uses ```amp.scale_loss``` to package the loss, so that apex can use loss's value for auto-precision tuning:

```
with amp.scale_loss(loss, optimizer) as scaled_loss:
   scaled_loss.backward()
```

To conclude, Apex's parallel training is related to:

```
# 4.apex_distributed2.py
import torch
import argparse
import torch.distributed as dist

from apex.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
torch.cuda.set_device(args.local_rank)

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

model = ...
model, optimizer = amp.initialize(model, optimizer)
model = DistributedDataParallel(model, device_ids=[args.local_rank])

optimizer = optim.SGD(model.parameters())

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      optimizer.zero_grad()
      with amp.scale_loss(loss, optimizer) as scaled_loss:
         scaled_loss.backward()
      optimizer.step()
```

Call ```torch.distributed.launch``` to launch the service:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 4.apex_distributed2.py
```

For full-version of using CIFAR10, click: [Github](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/4.apex_distributed2.py)。

## Horovod implementation

> Horovod is Uber's open-source deep learning package, which developed by inspiring Facebook "Training ImageNet In 1 Hour" and baidu's "Ring Allreduce". It is easy to use by using PyTorch/Tensorflow.

At API level, Horovod is alike ```torch.distributed```. Basing on ```mpirun```, Horovod uses its own ```horovodrun``` as the launcher.

Alike ```torch.distributed.launch```, we only need to write one part of code and ```horovodrun``` launcher will help us automatically assign it to each process which further run at each GPU. During executing, the launcher will transfer the current process's index (GPU's local rank) to hvd, so that we can obtain the index of current process: 

```
import horovod.torch as hvd

hvd.local_rank()
```

alike ```init_process_group```, Horovod  uses ```init``` to set up GPUs' communication backend and portation:
```
hvd.init()
```

After, use ```DistributedSampler``` to split the dataset. As former mentioned, it helps us split each batch into partitions and each GPU will deal with one partition:

```
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)
```

Then, use ```broadcast_parameters``` to package model's parameters, and broadcast the model's parameters from root_rank GPU to all other GPUs:

```
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
```

Then use ```DistributedOptimizer``` to update the existing optimizer. Which can perform all-redue of different GPUs' gradients. After all-reduce, the GPUs' gradients are the average of the gradients of each GPU before all-reduce:

```
hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), compression=hvd.Compression.fp16)
```
Finally, push the minibatch data into current GPU and then we only need to write normal forward/backward codes:
```
torch.cuda.set_device(args.local_rank)

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)
      ...
      output = model(images)
      loss = criterion(output, target)
      ...
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

To conclude, Horovod's related code are:

```
# 5.horovod_distributed.py
import torch
import horovod.torch as hvd

hvd.init()
torch.cuda.set_device(hvd.local_rank())

train_dataset = ...
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=..., sampler=train_sampler)

model = ...
model.cuda()

optimizer = optim.SGD(model.parameters())

optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

for epoch in range(100):
   for batch_idx, (data, target) in enumerate(train_loader):
       images = images.cuda(non_blocking=True)
       target = target.cuda(non_blocking=True)
       ...
       output = model(images)
       loss = criterion(output, target)
       ...
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
```

At running, call the launcher ```horovodrun```:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 -H localhost:4 --verbose python 5.horovod_distributed.py
```

Full code using CIFAR10 is here: [Github](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/5.horovod_distributed.py)。

In addition, full code using MNIST is here: [Github](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/5.2.horovod_pytorch_mnist.py)，Click [Github](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/5.2.run.mnist.sh) for the bash that runs the python code.


## GPU 集群上的分布式 [not tested yet!]

> Slurm，是一个用于 Linux 系统的免费、开源的任务调度工具。它提供了三个关键功能。第一，为用户分配资源(计算机节点)，以供用户执行工作。第二，它提供了一个框架，用于执行在节点上运行着的任务(通常是并行的任务)，第三，为任务队列合理地分配资源。如果你还没有部署 Slurm 可以按照笔者总结的[部署教程](https://zhuanlan.zhihu.com/p/149771261)进行部署。

通过运行 slurm 的控制命令，slurm 会将写好的 python 程序在每个节点上分别执行，调用节点上定义的 GPU 资源进行运算。要编写能被 Slurm 在 GPU 集群上执行的 python 分布式训练程序，我们只需要对上文中多进程的 DistributedDataParallel 代码进行修改，告诉每一个执行的任务（每个节点上的 python 程序），要用哪些训练哪一部分数据，反向传播的结果如何合并就可以了。

我们首先需要获得每个任务（对应每个节点）的基本信息，以便针对任务的基本信息处理其应当负责的数据。在使用 slurm 执行 srun python 代码时，python 可以从环境变量 os.environ 中获取当前 python 进程的基本信息：

```
import os
local_rank = os.environ['SLURM_PROCID'] # 当前任务的编号（比如节点 1 执行 1 号任务，节点 2 执行 2 号任务）
world_size = os.environ['SLURM_NPROCS'] # 共开启的任务的总数（共有 2 个节点执行了 2 个任务）
job_id = os.environ['SLURM_JOBID'] # 当前作业的编号（这是第 1 次执行 srun，编号为 1）
```

在每个任务（节点）中，我们需要为节点中的每个 GPU 资源分配一个进程，管理该 GPU 应当处理的数据。

当前节点的 GPU 的数量可以由 torch.cuda 查询得到：

```
ngpus_per_node = torch.cuda.device_count()
```

接着，与上文相似，我们使用 torch.multiprocessing 创建 ngpus_per_node 个进程，其中，每个进程执行的函数为 main_worker ，该函数调用所需要的由 args 传入：

```
mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
```

在编写 main_worker 时，我们首先需要解决的问题是：不同节点、或者同一节点间的不同进程之间需要通信来实现数据的分割、参数的合并。我们可以使用 pytorch 的 dist 库在共享文件系统上创建一个文件进行通信：

```
import torch.distributed as dist

def main_worker(gpu, ngpus_per_node, args):
  dist_url = "file://dist_file.{}".format(job_id)
  rank = local_rank * ngpus_per_node + gpu
  dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
  ...
```

完成进程创建和通信后，下一步就是实现我们常用的 pipline 了，即加载模型、加载数据、正向传播、反向传播。与上文相似，这里，我们把模型加载进当前进程所对应的 GPU 中：

```
def main_worker(gpu, ngpus_per_node, args):
  dist_url = "file://dist_file.{}".format(job_id)
  rank = local_rank * ngpus_per_node + gpu
  dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
  ...
  torch.cuda.set_device(gpu)
  model.cuda(gpu)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
```

接着，把当前进程对应的数据段采样出来，也加载到对应的 GPU 中。同样可以使用 pytorch 的 dist 库实现这个采样过程：

```
def main_worker(gpu, ngpus_per_node, args):
  dist_url = "file://dist_file.{}".format(job_id)
  rank = local_rank * ngpus_per_node + gpu
  dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
  ...
  torch.cuda.set_device(gpu)
  model.cuda(gpu)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
  ...
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=2,
                                             pin_memory=True,
                                             sampler=train_sampler)
  for i, (images, target) in enumerate(train_loader):
    images = images.cuda(gpu, non_blocking=True)
    target = target.cuda(gpu, non_blocking=True)
```

最后，进行正常的正向和反向传播：

```
def main_worker(gpu, ngpus_per_node, args):
  dist_url = "file://dist_file.{}".format(job_id)
  rank = local_rank * ngpus_per_node + gpu
  dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
  ...
  torch.cuda.set_device(gpu)
  model.cuda(gpu)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
  ...
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=2,
                                             pin_memory=True,
                                             sampler=train_sampler)
  for i, (images, target) in enumerate(train_loader):
    images = images.cuda(gpu, non_blocking=True)
    target = target.cuda(gpu, non_blocking=True)
    ...
    output = model(images)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

在使用时，调用 srun 启动任务：

```
srun -N2 --gres gpu:1 python 6.distributed_slurm_main.py --dist-file dist_file
```

在 ImageNet 上的完整训练代码，请点击[Github](https://github.com/Xianchao-Wu/pytorch-distributed/blob/master/6.distributed_slurm_main.py)。

## 分布式 evaluation

> all_reduce, barrier 等 API 是 distributed 中更为基础和底层的 API。这些 API 可以帮助我们控制进程之间的交互，控制 GPU 数据的传输。在自定义 GPU 协作逻辑，汇总 GPU 间少量的统计信息时，大有用处。熟练掌握这些 API 也可以帮助我们自己设计、优化分布式训练、测试流程。

到目前为止，Distributed Sampler 能够帮助我们分发数据，DistributedDataParallel、hvd.broadcast_parameters 能够帮助我们分发模型，并在框架的支持下解决梯度汇总和参数更新的问题。然而，还有一些同学还有这样的疑惑，

1. 训练样本被切分成了若干个部分，被若干个进程分别控制运行在若干个 GPU 上，如何在进程间进行通信汇总这些（GPU 上的）信息？
2. 使用一张卡进行推理、测试太慢了，如何使用 Distributed 进行分布式地推理和测试，并将结果汇总在一起？
3. ......

要解决这些问题，我们缺少一个更为基础的 API，**汇总记录不同 GPU 上生成的准确率、损失函数等指标信息**。这个 API 就是 `torch.distributed.all_reduce`。示意图如下：

![img](https://pic4.zhimg.com/80/v2-f424bdc8108abd5421e3af3b902b2ccf_720w.jpg)

具体来说，它的工作过程包含以下三步：

1. 通过调用 `all_reduce(tensor, op=...)`，当前进程会向其他进程发送 `tensor`（例如 rank 0 会发送 rank 0 的 tensor 到 rank 1、2、3）
2. 接受其他进程发来的 `tensor`（例如 rank 0 会接收 rank 1 的 tensor、rank 2 的 tensor、rank 3 的 tensor）。
3. 在全部接收完成后，当前进程（例如rank 0）会对当前进程的和接收到的 `tensor` （例如 rank 0 的 tensor、rank 1 的 tensor、rank 2 的 tensor、rank 3 的 tensor）进行 `op` （例如求和）操作。

使用 `torch.distributed.all_reduce(loss, op=torch.distributed.reduce_op.SUM)`，我们就能够对不数据切片（不同 GPU 上的训练数据）的损失函数进行求和了。接着，我们只要再将其除以进程（GPU）数量 `world_size`就可以得到损失函数的平均值。

正确率也能够通过同样方法进行计算：

```
# 原始代码
output = model(images)
loss = criterion(output, target)
        
acc1, acc5 = accuracy(output, target, topk=(1, 5)) # 对于CIFAR10，没有使用top5的结果，只有top1的结果，top5.acc=top1.acc.
losses.update(loss.item(), images.size(0))
top1.update(acc1.item(), images.size(0))
top5.update(acc5.item(), images.size(0))
​
# 修改后，同步各 GPU 中数据切片的统计信息，用于分布式的 evaluation
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt
​
output = model(images)
loss = criterion(output, target)
acc1, acc5 = accuracy(output, target, topk=(1, 5))
​
torch.distributed.barrier()
​
reduced_loss = reduce_tensor(loss.data)
reduced_acc1 = reduce_tensor(acc1)
reduced_acc5 = reduce_tensor(acc5)
​
losses.update(loss.item(), images.size(0))
top1.update(acc1.item(), images.size(0))
top5.update(acc5.item(), images.size(0))
```

值得注意的是，为了同步各进程的计算进度，我们在 reduce 之前插入了一个同步 API `torch.distributed.barrier()`。在所有进程运行到这一步之前，先完成此前代码的进程会等待其他进程。这使得我们能够得到准确、有序的输出。在 Horovod 中，我们无法使用 `torch.distributed.barrier()`，取而代之的是，我们可以在 allreduce 过程中指明：

```
def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    hvd.allreduce(rt, name='barrier')
    #rt /= world_size # attention: do not /= world_size here for horovod/hvd! since allreduce already includes average!
    return rt
    
output = model(images)
loss = criterion(output, target)
acc1, acc5 = accuracy(output, target, topk=(1, 5))

reduced_loss = reduce_tensor(loss.data)
reduced_acc1 = reduce_tensor(acc1)
reduced_acc5 = reduce_tensor(acc5)

losses.update(loss.item(), images.size(0))
top1.update(acc1.item(), images.size(0))
top5.update(acc5.item(), images.size(0))
```
