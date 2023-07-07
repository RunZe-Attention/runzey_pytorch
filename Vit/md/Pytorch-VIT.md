# <font color = 'red'>Vision Transformer</font>

- Encoder only

## Transformer应用于CV的挑战

- 1.如NLP一样将单个word作为一个token,因为句子长度不一定很长,所以计算量还可以hold住,但是如果输入的是图片数据,单个像素点作为一个token,将导致计算量异常的大
- 2.并且单个pixel不会想一个word一样能够表达很多信息,大部分时间单个pixel意义是不大的,所以并不能已单个pixel作为一个token看待
- 3.综上,不能将Trn应用到pixel级别做attention











