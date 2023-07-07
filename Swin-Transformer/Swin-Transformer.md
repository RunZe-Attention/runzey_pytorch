# <font color = 'red'>分享目的</font>

- Swin-Transformer崛起，剑指CV通用模型(分类,检测,分割等...)
- CV团队首选
- 希望这次分享能为大家理清Swin整体脉络



# <font color = 'red'>本文将从以下几个方面介绍Swin-Transformer</font>

- 0.整体预览

- 1.基于图片生成的Patch embbeding
- 2.构建MHSA并计算其时间复杂度
- 3.构建Window-MHSA并计算其时间复杂度(即为什么不会像VIT一样,计算全局自注意力)
- 4.构建Shifted-Window-MHSA 及其mask矩阵
- 5.构建Patch Merging
- 6.构建Swin-Transformer Block
- 7.构建Swin-Transformer Model
- 8.测试demo



## 0.整体预览



![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/整体架构.png)

![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/compare.png)



### (a) Architecture代表网络的整体

### (b) Two Successsive Swin Transformer Blocks 即左图中的每个Stage的Swin Transformer Block

### (a) Architecture

- 0.输入image的形状为$H \times W \times 3$
- 1.图片在进入Stage1之前,使用patch size = 4超参，将整幅图片划分为 $\frac{H}{4} \times \frac{W}{4} $ 个patch,并且每个patch的depth变为$4 \times4 \times3$
- 2.在Stage1中的Linear Embbeding中将depth为48 通过一次线性映射,使得depth被映射为长度为C的tensor,之后经历了Swin Tranformer Block,这里的Transformer与其他模型使用的Transformer一样，只是单纯的计算词与词之间,或者的patch与patch之间的自注意力得分，不会对输入进行任何shape上的改变。
- 3.进入stage2 首先会经过Patch Merging layer ,这有些类似于CNN网络,将多个patch进一步的进行merge,图示所选取的merge size=2合并为一个window,即会有2*2个patch被合并为一个window,注意此时特征图的尺寸缩小了四倍(即H与W均缩小2倍),而depth也应该扩充为输入时的四倍，原文的做法是将输出的mode_dim_C做一次线性变换，总4倍特征维度映射为2倍特征维度。
- Stage3与Stage4采取了与Stage2同样的策略。
- 完成模型的前向过程之后,原本没有继续进行模型的推演,原因也与前面说过,Swin将作为一种通用模型，在Stage4得到的特征图可以继续进行主流任务，包括分类、检测、分割等。

### (b) Two Successsive Swin Transformer Blocks 

- Swin Transformer block主要由两部分构成:W-MSA 与 SW-MSA
- 左侧以W-MSA为主导,依次经过了Layer norm、W-MSA(此处做了残差连接),之后再次经过Layer norm 以及MLP(注意Transformer的MLP一般分为两部分，第一阶段将特征维度映射到更高维度空间中,通常为4倍关系,第二阶段再次映射到输入维度空间)，此处再次进行了残差连接
- 右侧结构基本与左侧一致，差别在于将W-MSA换位SW-MSA模块
- W-MSA与SW-MSA作为Swin Transformer最大亮点,将在代码中做详细说明



## <font color = 'red'>1、如何基于图片生成patch_embbeding</font>

- 基于pytorch的unfold API来将图片进行分patch,也就是模仿卷积的思路，设置kernel_size=stride=patch_size,得到分块后的图片
- 得到的图片格式为[bs,num_patch,patch_depth]
- 将上述张量与形状为[patch_depth,model_dim_C]的张量做线性映射，即可得到[bs,num_patch,model_dim_C]的patch_embbeding

```PYTHON
def image2emb_naive(image,patch_size,weight):
  # [bs,num_patch,all_depth]
    patch=F.unfold(image,kernel_size=(patch_size,patch_size)，stride = (patch_size,patch_size)).transpose(-1,-2)
    patch_embbeding = patch @ weight
    return patch_embbeding

image = torch.randn(1,3,8,8)
weight = torch.randn(4*4*3,8)

print(image2emb_naive(image,4,weight).shape)
```

## 2、构建MHSA并计算其复杂度

- 1.基于输入shape:[BS,L,C]进行三个映射分别得到q、k、v三个矩阵
  - 则每个矩阵q、k、v 的计算复杂度为$LC^2$,一共是$3LC^2$
- 2.计算attention时候的复杂度
  - 1.$q@k^T$的复杂度为$L^2C$
  - 2.继续乘以v的复杂度为$L^2C$
  - 3.最后做一个线性映射的复杂度为$LC^2$
- 3.所以MHSA的时间复杂度为
  - $4LC^2+2L^2C$
- 4.可以看到MHSA有着与L(seq_len 或者 num_patch)的平方的关系,具有很高的时间复杂度



```PYTHON
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,model_dim,num_head):
        super(MultiHeadSelfAttention,self).__init__()
        self.num_head = num_head

        self.proj_linear_layer = nn.Linear(model_dim, 3*model_dim)
        self.final_linear_layer = nn.Linear(model_dim, model_dim)

    def forward(self, input, additive_mask=None):
        bs,seq_len,model_dim = input.shape
        num_head = self.num_head
        
        head_dim = model_dim // num_head
        
        proj_output = self.proj_linear_layer(input)
        q, k, v = proj_output.chunk(3,dim=-1) # shape均为[bs,seq_len,model_dim_C]
        # q:[bs,seq_len,model_dim]
        q = q.reshape(bs, seq_len, num_head, head_dim).transpose(1, 2).reshape(bs * num_head, seq_len, head_dim)
        k = k.reshape(bs, seq_len, num_head, head_dim).transpose(1, 2).reshape(bs * num_head, seq_len, head_dim)
        v = v.reshape(bs, seq_len, num_head, head_dim).transpose(1, 2).reshape(bs * num_head, seq_len, head_dim)
        if additive_mask is None:
            attention_prob = F.softmax(torch.bmm(q,k.transpose(-1,-2))/math.sqrt(head_dim),dim=-1)
        else:
            additive_mask = additive_mask.tile(num_head,1,1)
            attention_prob = F.softmax(torch.bmm(q,k.transpose(-1,-2))/math.sqrt(head_dim)+additive_mask,dim=-1)
        output = torch.bmm(attention_prob, v) # [bs*num_head,seq_len,head_dim]
        output = output.reshape(bs, num_head, seq_len, head_dim).transpose(1,2)
        output = output.reshape(bs, seq_len, model_dim)
        output = self.final_linear_layer(output)
        return attention_prob, output 
```



## 3、构建Window MHSA 并计算其复杂度

- 将patch后的图片进一步分成一个个更大的window
  - 1.需要将3D的patch embedding转换为图片的格式
  - 2.使用unfold函数将patch划分为window
- 在每个window内部计算MHSA
  - window数目可以跟batch_size进行统一对待,因为在window之间没有交互计算
  - 关于计算W-MHSA的时间复杂度
    - 假设window的边长为W,那么计算每个窗的复杂度为$4W^2C^2+2W^4C$
    - window的个数为$\frac{L}{W^2}$
    - 因此总的复杂度为二者相乘$4LC^2+2LW^2C$
  - 此处不需要mask
  - 将计算结果转换成带window的4D tenser

- 复杂度对比
  - `MHSA`: $4LC^2+2L^2C$
  - `W-MHSA`: $4LC^2+2LW^2C$

```PYTHON
def windows_multi_head_self_attention(patch_embbeding,mhsa,window_size=4,num_head=2):
    # [bs,num_window,num_patch_in_window,patch_depth]
    num_patch_in_window = window_size*window_size
    bs, num_patch, patch_depth = patch_embbeding.shape
    image_height = image_width = int(math.sqrt(num_patch))

    patch_embbeding = patch_embbeding.transpose(-1, -2)
    patch = patch_embbeding.reshape(bs,patch_depth, image_height, image_width)
    window=F.unfold(patch,kernel_size=(window_size,window_size),stride=(window_size,window_size),).transpose(-1,-2)


    bs,num_windows,patch_depth_times_num_patch_in_window = window.shape
    window = window.reshape(bs*num_windows,patch_depth,num_patch_in_window).transpose(-1,-2)

    _, output = mhsa(window)# [bs*num_window,num_patch_in_window,patch_depth]

    output = output.reshape(bs,num_windows,num_patch_in_window,patch_depth)
    return output
```

## 4、构建shifted Window MHSA 及其MASK

- 将上一步的 W-MHSA转换为图片的形式即:[bs,num_windows,num_patch_in_window,patch_depth]转换为[bs,ic,image_h,image_w]
- `假设`已经做了新的window划分,这一步叫做shift-window
- 为了保持window数目不变从而有高效的计算,需要将图片的patch往左和往上各自滑动半个窗口大小的步长,保持patch所属window类别不变
- 将图片patch还原成window的数据格式
- 由于shift_window之后,每个window虽然形状一致,但是部分window存在不属于同一个窗口的patch,所以要生成mask
- 如何生成mask
  - 首先构建一个shift-window的patch所属的window类别矩阵
  - 对该矩阵进行同样的王座和网上各自滑动半个窗口大小的步长的操作
  - 通过unfold操作可得到[bs,num_window,num_patch_in_window]形状的类别矩阵
  - 对该矩阵进行扩维:[bs,num_window,num_patch_in_window,1]
  - 该矩阵与自身的转置作差,得到同类关系矩阵，（为0的位置patch的关系属于同类）
  - 对同类矩阵中的非0的位置用负无穷进行填充,对于零的位置用0去填充,这样就构建好了MHS所需要的Mask
  - 这个mask矩阵的形状为[bs,num_window,num_patch_in_window,num_patch_in_window]
- 将window转换为三维的形式:[bs * num_window,num_patch_in _window，patch_size]
- 将三维格式的特征连通mask一起送入MHSA中计算得到注意力输出
- 将注意力输出换转为图片patch形式:[bs,num_window,num_patch_in_window,patch_size]
- 为了恢复位置,需要将图片的patch往右和往下滑动半个window大小
- 至此,SW-MHSA计算完成



![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/shift-window.png)



关于shift_window具体的过程

![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/SW-MHSA.png)



![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/one.png)

![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/two.png)



![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/three.png)

![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/four.png)

![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/five.png)

```PYTHON
def shift_window_multi_head_self_attention(w_msa_output,mhsa,window_size=4,num_head=2):
    bs,num_window,num_patch_in_window,patch_depth = w_msa_output.shape
    shifted_w_msa_output ,additive_mask = shift_window(w_msa_output,window_size,
                                                       shift_size=-window_size//2,generate_mask=True)

    shifted_w_msa_output = shifted_w_msa_output.reshape(bs*num_window,num_patch_in_window,patch_depth)

    _, output = mhsa(shifted_w_msa_output,additive_mask)
    output = output.reshape(bs, num_window,num_patch_in_window,patch_depth)

    output,_ = shift_window(output,window_size,shift_size=window_size//2,generate_mask=False)
    return output
```

- window格式转换为image格式

```PYTHON
def window2image(msa_output):
    bs, num_windows, num_patch_in_windows, patch_depth = msa_output.shape
    window_size = int(math.sqrt(num_patch_in_windows))
    image_height = image_width = int(math.sqrt(num_windows)) * window_size
    msa_output = msa_output.reshape(bs,int(math.sqrt(num_windows)),int(math.sqrt(num_windows)),window_size,window_size,patch_depth)
    msa_output = msa_output.transpose(2,3)
    image = msa_output.reshape(bs,image_height*image_width,patch_depth)
    image = image.transpose(-2,-1)
    image = image.reshape(bs,patch_depth,image_height,image_width)
    return image
```

```python
def shift_window(w_msa_output,window_size,shift_size,generate_mask=False):
    bs, num_window, num_patch_in_window,patch_depth = w_msa_output.shape
    w_msa_output = window2image(w_msa_output)

    bs,patch_depth,image_height,image_weight = w_msa_output.shape

    rolled_w_msa_output = torch.roll(w_msa_output,shifts=(shift_size,shift_size),dims=(2,3))

    shifted_w_msa_output = rolled_w_msa_output.reshape(bs,patch_depth,int(math.sqrt(num_window)),window_size,int(math.sqrt(num_window)),window_size)

    shifted_w_msa_output = shifted_w_msa_output.transpose(3,4)
    shifted_w_msa_output = shifted_w_msa_output.reshape(bs,patch_depth,num_window*num_patch_in_window)
    shifted_w_msa_output = shifted_w_msa_output.transpose(-1,-2)
    shifted_window = shifted_w_msa_output.reshape(bs,num_window,num_patch_in_window,patch_depth)

    if generate_mask:
        additive_mask = build_mask_for_shifted_wmsa(bs,image_height,image_weight,window_size)
    else:
        additive_mask = None

    return shifted_window,additive_mask
```

```PYTHON
def build_mask_for_shifted_wmsa(batch_size,image_height,image_width,window_size):
    index_matrix = torch.zeros(image_height,image_width)
    for i in range(image_height):
        for j in range(image_width):
            row_times = (i+window_size//2) // window_size
            col_times = (j+window_size//2) // window_size
            value = row_times * (image_height // window_size) + col_times + 1
            index_matrix[i,j] = value

    roll_index_matrix = torch.roll(index_matrix,shifts=(-window_size//2,-window_size//2),dims=(0,1))
    roll_index_matrix = roll_index_matrix.unsqueeze(0).unsqueeze(0) #[bs,ch,h,w]

    c=F.unfold(roll_index_matrix,kernel_size=(window_size,window_size),stride=(window_size,window_size)).transpose(-1,-2)

    #c = c.tile(batch_size,1,1)

    bs,num_window,num_patch_in_window = c.shape

    c1 = c.unsqueeze(-1)
    c2 = (c1-c1.transpose(-1,-2)) == 0
    valid_matrix = c2.to(torch.float32)

    unlimit_min = -1e-9
    additive_mask = (1-valid_matrix) * unlimit_min

    additive_mask = additive_mask.reshape(bs * num_window,num_patch_in_window,num_patch_in_window)
    return additive_mask
```





## 5、如何构建PatchMerging

- 将window转换为patch
- 利用unfold操作,按照merge_size*merge_size大小得到新的patch,[bs,num_patch_new,merge_size*merge_size*patch_depth_old]
- 增加全连接层对patch_depth进行映射,一般将维度成0.5倍,输出的patch_embbe的形状为[bs,num_patch,patch_depth]
- 举例说明:以merge_size=2为例,num_patch变为原来的1/4,patch_depth变为原来的2倍

```PYTHON
class PatchMerging(nn.Module):
    def  __init__(self, model_dim, merge_size, output_depth_scale = 0.5):
        super(PatchMerging,self).__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(
            model_dim*merge_size*merge_size,
            int(model_dim*merge_size*merge_size*output_depth_scale))

    def forward(self,input):
        bs,num_window,num_patch_in_window,patch_depth = input.shape
        #window_size = int(math.sqrt(num_patch_in_window))
        input = window2image(input)  # [bs,patch_depth,image_h,image_w]
        merged_window = F.unfold(input,kernel_size=(self.merge_size,self.merge_size),
                                 stride=(self.merge_size,self.merge_size)).transpose(-1,-2)

        merged_window = self.proj_layer(merged_window)

        return merged_window
```





## 6、构建SwinTransformerBlock

![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/整体架构.png)

- 1、每个block包含有LayerNorm,WMHSA、MLP、SMHSA
- 2、输入的是patch_embbeding[bs,num_patch,patch_depth]
- 3、其中每个MLP与所有的Transformer族一致，都是包含了两个Linear,第一个Layer将model_dim_C映射到4*model_dimC,第二个layer将4*model_dim_C映射会model_dim_C维度
- 4、输出的维度为[bs,num_patch,num_patch_in_window,patch_depth]
- 5、注意残差链接的时候输入与输出的维度需要保持一致

```python
class SwinTransformerBlock(nn.Module):
    def __init__(self,model_dim,window_size,num_head):
        super(SwinTransformerBlock,self).__init__()
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)
        self.layer_norm4 = nn.LayerNorm(model_dim)

        self.wsma_mlp1 = nn.Linear(model_dim,4*model_dim)
        self.wsma_mlp2 = nn.Linear(4*model_dim,model_dim)
        
        self.swsma_mlp1 = nn.Linear(model_dim, 4 * model_dim)
        self.swsma_mlp2 = nn.Linear(4 * model_dim, model_dim)

        self.mhsa1 = MultiHeadSelfAttention(model_dim,num_head)
        self.mhsa2 = MultiHeadSelfAttention(model_dim,num_head)

    def forward(self, input):
        bs,num_patch,patch_depth = input.shape
        
        # WMHSA
        input1 = self.layer_norm1(input)
        w_msa_output = windows_multi_head_self_attention(input1,self.mhsa1,window_size=4,num_head=2)
        bs,num_window, num_patch_in_window, patch_depth = w_msa_output.shape
        w_msa_output = input1 + w_msa_output.reshape(bs,num_patch,patch_depth)
        output1 = self.wsma_mlp2(self.wsma_mlp1(self.layer_norm2(w_msa_output)))
        output1 += w_msa_output

        # SWMHSA
        input2 = self.layer_norm3(output1)
        input2 = input2.reshape(bs,num_window,num_patch_in_window,patch_depth)
        sw_msa_output = shift_window_multi_head_self_attention(input2, self.mhsa2,window_size=4,num_head=2)
        sw_msa_output = output1+sw_msa_output.reshape(bs,num_patch,patch_depth)
        output2 = self.swsma_mlp2(self.swsma_mlp1(self.layer_norm4(sw_msa_output)))
        output2 += sw_msa_output

        output2 = output2.reshape(bs,num_window,num_patch_in_window,patch_depth)
        return output2
```

## 7、构建SwinTransformerModel

![整体架构](/Users/yangrunze/Desktop/git_pt/swin-pic/整体架构.png)



- 输入是图片[bs,ic,image_h,image_w]
- 首先对图片进行分块得到patch embedding
- 根据论文进入四个stage
- 对最后一个输出转换为patch embedding的形式[bs,num_patch,patch_depth]
- 对patch embbeding进行时间维度的平均池化操作,并映射得到分类的logits,分类完毕

```python
class SwinTransformerModel(nn.Module):
    def __init__(self,input_image_channel = 3,patch_size = 4,model_dim_C = 8,num_classes = 10,window_size = 4,num_head=2,merge_size = 2):
        super(SwinTransformerModel,self).__init__()

        patch_depth = patch_size*patch_size*input_image_channel
        self.patch_size = patch_size
        self.model_dim_C = model_dim_C
        self.num_classes = num_classes

        self.patch_embbeding_weight = nn.Parameter(torch.randn(patch_depth,model_dim_C))

        self.block1 = SwinTransformerBlock(model_dim_C,window_size,num_head)
        self.block2 = SwinTransformerBlock(model_dim_C*2, window_size, num_head)
        self.block3 = SwinTransformerBlock(model_dim_C*4, window_size, num_head)
        self.block4 = SwinTransformerBlock(model_dim_C*8, window_size, num_head)

        self.patch_merging1 = PatchMerging(model_dim_C,merge_size)
        self.patch_merging2 = PatchMerging(model_dim_C*2, merge_size)
        self.patch_merging3 = PatchMerging(model_dim_C*4, merge_size)

        self.final_linear = nn.Linear(model_dim_C*8,num_classes)


    def forward(self,image):
        patch_embbeding_naive = image2emb_naive(image,self.patch_size,self.patch_embbeding_weight)

        # stage1
        patch_embbeding = patch_embbeding_naive
        print(patch_embbeding.shape)
        sw_msa_output = self.block1(patch_embbeding)
        print("stage1_output:",sw_msa_output.shape)

        # stage2
        merged_patch1 = self.patch_merging1(sw_msa_output)
        sw_msa_output_1 = self.block2(merged_patch1)
        print("stage2_output:", sw_msa_output_1.shape)

        # stage3
        merged_patch2 = self.patch_merging2(sw_msa_output_1)
        sw_msa_output_2 = self.block3(merged_patch2)
        print("stage3_output:", sw_msa_output_2.shape)

        # stage4
        merged_patch3 = self.patch_merging3(sw_msa_output_2)
        sw_msa_output_3 = self.block4(merged_patch3)
        print("stage4_output:", sw_msa_output_3.shape)

        # logits
        bs,num_window,num_patch_in_window,patch_depth = sw_msa_output_3.shape
        
        sw_msa_output_3 = sw_msa_output_3.reshape(bs,-1,patch_depth)
        pool_output = torch.mean(sw_msa_output_3,dim=1)[bs,patch_depth]
        logits = self.final_linear(pool_output)

        print("logits",logits.shape)
```

