# 卷积转置

- 也不考虑batch_size以及输入channel

- 代码:

  ```PYTHON
  import torch
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import math
  
  def get_kernel_matrix(kernel,input_size):
      kernel_h,kernel_w = kernel.shape
      input_h,input_w = input_size
      num_out_feature_map = (input_h-kernel_h+1) * (input_w-kernel_w+1)
      result = torch.zeros((num_out_feature_map,input_h * input_w))
  
      row_index = 0
      for i in range(0,input_h - kernel_h + 1 , 1):
          for j in range(0,input_w - kernel_w + 1,1):
              kernel_pad = F.pad(kernel,(i,input_h-kernel_h-i,j,input_w-kernel_w-j))
              result[row_index] = kernel_pad.flatten()
              row_index+=1
  
      return result
    
  
  kernel = torch.randn(3,3)
  input = torch.randn(4,4)
  
  kernel_matrix= get_kernel_matrix(kernel,input.shape)
  diy = kernel_matrix@input.reshape((-1,1))
  api = F.conv2d(input.unsqueeze(0).unsqueeze(0),kernel.unsqueeze(0).unsqueeze(0))
  
  ## 做转置
  diy=kernel_matrix.transpose(-1,-2) @ diy
  api=F.conv_transpose2d(api,kernel.unsqueeze(0).unsqueeze(0))
  
  print(diy.reshape(4,4))
  print(api)
  
  ```



- 输出

  ```python
  tensor([[-1.9440,  2.6080, -0.1282,  0.0442],
          [-2.6682,  6.9085, -2.1691,  0.6611],
          [ 0.3145,  0.7952, -5.2570,  2.8353],
          [-0.5514,  0.9517, -1.4789,  3.6293]])
  tensor([[[[-1.9440,  2.6080, -0.1282,  0.0442],
            [-2.6682,  6.9085, -2.1691,  0.6611],
            [ 0.3145,  0.7952, -5.2570,  2.8353],
            [-0.5514,  0.9517, -1.4789,  3.6293]]]])
  ```

  