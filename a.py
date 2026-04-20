# from collections import OrderedDict

# x=OrderedDict({
#             'STOP': [0],
#             "MOVE_FORWARD": [1],
#             "TURN_LEFT": [2],
#             "TURN_RIGHT": [3]
#         })

# a='S TOP'

# if a in x:
#     print(x[a])
    
# a='sTOP'
# if a in x:
#     print(x[a])


import torch

seq=5

pad_id=0

print(seq!=pad_id)

seqence=torch.tensor([5,0,3,0,2])
mask=seqence!=pad_id
print(mask)


def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones(size, size)).type(torch.bool)  # 下三角矩阵
    return mask

mask=create_look_ahead_mask(5)

print(mask)

x=torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
y=torch.arange(0,5)
out=x+y
print(x.shape,y.shape)
print(out)