import torch
input = [
    [2, 3, 4, 5, 0, 0],
    [1, 4, 3, 0, 0, 0],
    [4, 2, 2, 5, 7, 0],
    [1, 0, 0, 0, 0, 0]
]
input = torch.tensor(input)
length = torch.LongTensor([[3], [2], [4], [0]])
# index之所以减1，是因为序列维度从0开始计算的
out = torch.gather(input, 1, length)
print(out)

a = torch.tensor([[1.3, 100, 3, 4.5, 5],
                  [2.0, 3, 0.3, 4.1, 2],
                  [6, 7, 8, 9, 2],
                  [10, 5, 0, 6, 8]])
b = torch.tensor([[1],
                  [2],
                  [3],
                  [0]])
c = torch.gather(a, 0, b)    # 输出a中第1维索引分别是1，2，3，4的元素：2，0.3，9，8
print(c)


