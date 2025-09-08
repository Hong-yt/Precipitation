import torch
print(torch.__version__)
print(torch.version.cuda)#cuda版本
print(torch.backends.cudnn.version())
print(torch.cuda.is_available()) #cuda是否可用，返回为True表示可用
print(torch.cuda.device_count())#返回GPU的数量
print(torch.cuda.get_device_name(1))#返回gpu名字，设备索引默认从0开始

