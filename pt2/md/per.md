

# Video Swin Transformer Large与Bert Base做Meter fusion



## 一、模型背景

## Meter模型由Bytedance AML向Graphcore发起需求,目前应用于视频审核业务中。Graphcore已驻场完成适配，性能提升明显。



## 二、当前性能以及与GPU比较

| 硬件平台 | Batch size | Throughput(samples/s) | Latency(ms) |
| :------: | :--------: | :-------------------: | :---------: |
|   C600   |     1      |          117          |    8.572    |
|   A30    |     1      |          31           |     33      |
|    T4    |     1      |           3           |      /      |

 





#### 说明 : C600与A30性能由Graphcore实测得出。由于客户要求希望C600性能强于A30,所以T4数据并未进行实测,当前数据由Bytedance AML团队提供。



