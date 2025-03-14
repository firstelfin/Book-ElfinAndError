# 使用高维张量索引高维张量

参考链接：[torch_indexed with multiple tensors](https://stackoverflow.com/questions/62241956/pytorch-index-high-dimensional-tensor-with-two-dimensional-tensor)

问题来源于YOLOV9代码解析：
```python
def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):

        gt_labels = gt_labels.to(torch.long)  # b, max_num_obj, 1; max_num_obj是当前batch标注最多的图片对应的实例数量
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # get the scores of each grid for each gt cls
        bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, h*w

        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(0)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps
```

对于一个高维张量，在某个特定维度使用一维张量索引，这个操作比较常规了；对多个轴使用一维张量索引也很常见了，下面我们将分析两种场景：
1. 索引张量时，某个轴使用高维张量索引；
2. 索引张量时，多个轴使用高维张量索引；

## 1、某个轴使用高维张量索引

### 1.1 生成基础张量

```python
import torch

pd_scores = torch.randn((2,3,4))

# batch_size = 2
# total_predict_num = 3
# class_num = 4
# pd_scores = tensor(
#     [[[ 0.8385, -1.0320, -0.4704,  0.2565],
#       [ 1.7174, -0.5995,  0.7012,  0.7073],
#       [ 0.0082,  0.0642, -0.2494, -0.7311]],

#      [[-0.1341,  0.6296,  0.6135,  2.6176],
#       [-1.5869,  0.0341, -0.9411,  1.2330],
#       [-0.8380,  0.5051, -0.1915, -0.0850]]])
```

### 1.2 生成索引张量

```python
ind = torch.zeros([2, 2, 5], dtype=torch.long)  # 索引必须时long，不然会报错
ind[0] = torch.arange(end=2).view(-1, 1).repeat(1, 5)
ind[1][0][0] = 1
ind[1][0][1] = 3
ind[1][1][0] = 1

# >>> ind
# tensor([[[0, 0, 0, 0, 0],    # 标识批量维度的索引
#          [1, 1, 1, 1, 1]],

#         [[1, 3, 0, 0, 0],    # 标识标注类别id的索引
#          [1, 0, 0, 0, 0]]])
```

### 1.3 在某个轴上使用高维张量索引

第一个维度索引取值
```python
>>> pd_scores[ind[0], :, :]
tensor([[[[ 0.8385, -1.0320, -0.4704,  0.2565],
          [ 1.7174, -0.5995,  0.7012,  0.7073],
          [ 0.0082,  0.0642, -0.2494, -0.7311]],

         [[ 0.8385, -1.0320, -0.4704,  0.2565],
          [ 1.7174, -0.5995,  0.7012,  0.7073],
          [ 0.0082,  0.0642, -0.2494, -0.7311]],

         [[ 0.8385, -1.0320, -0.4704,  0.2565],
          [ 1.7174, -0.5995,  0.7012,  0.7073],
          [ 0.0082,  0.0642, -0.2494, -0.7311]],

         [[ 0.8385, -1.0320, -0.4704,  0.2565],
          [ 1.7174, -0.5995,  0.7012,  0.7073],
          [ 0.0082,  0.0642, -0.2494, -0.7311]],

         [[ 0.8385, -1.0320, -0.4704,  0.2565],
          [ 1.7174, -0.5995,  0.7012,  0.7073],
          [ 0.0082,  0.0642, -0.2494, -0.7311]]],


        [[[-0.1341,  0.6296,  0.6135,  2.6176],
          [-1.5869,  0.0341, -0.9411,  1.2330],
          [-0.8380,  0.5051, -0.1915, -0.0850]],

         [[-0.1341,  0.6296,  0.6135,  2.6176],
          [-1.5869,  0.0341, -0.9411,  1.2330],
          [-0.8380,  0.5051, -0.1915, -0.0850]],

         [[-0.1341,  0.6296,  0.6135,  2.6176],
          [-1.5869,  0.0341, -0.9411,  1.2330],
          [-0.8380,  0.5051, -0.1915, -0.0850]],

         [[-0.1341,  0.6296,  0.6135,  2.6176],
          [-1.5869,  0.0341, -0.9411,  1.2330],
          [-0.8380,  0.5051, -0.1915, -0.0850]],

         [[-0.1341,  0.6296,  0.6135,  2.6176],
          [-1.5869,  0.0341, -0.9411,  1.2330],
          [-0.8380,  0.5051, -0.1915, -0.0850]]]])
```

最后一个维度索引取值
```python
>>> pd_scores[:, :, ind[1]]
tensor([[[[-1.0320,  0.2565,  0.8385,  0.8385,  0.8385],
          [-1.0320,  0.8385,  0.8385,  0.8385,  0.8385]],

         [[-0.5995,  0.7073,  1.7174,  1.7174,  1.7174],
          [-0.5995,  1.7174,  1.7174,  1.7174,  1.7174]],

         [[ 0.0642, -0.7311,  0.0082,  0.0082,  0.0082],
          [ 0.0642,  0.0082,  0.0082,  0.0082,  0.0082]]],


        [[[ 0.6296,  2.6176, -0.1341, -0.1341, -0.1341],
          [ 0.6296, -0.1341, -0.1341, -0.1341, -0.1341]],

         [[ 0.0341,  1.2330, -1.5869, -1.5869, -1.5869],
          [ 0.0341, -1.5869, -1.5869, -1.5869, -1.5869]],

         [[ 0.5051, -0.0850, -0.8380, -0.8380, -0.8380],
          [ 0.5051, -0.8380, -0.8380, -0.8380, -0.8380]]]])
```

通过上面的案例我们可以发现，shape的变化为：shape(2,3,4)->shape(2,5,3,4)、shape(2,3,4)->shape(2,3,2,5).
通过观察我们知道这里的高维索引，实际上是在索引张量上按照索引张量的shape遍历pd_scores对应维度进行取值。

## 2、多个轴使用高维张量索引

```python
>>> pd_scores[ind[0], :, ind[1]]
tensor([[[-1.0320, -0.5995,  0.0642],
         [ 0.2565,  0.7073, -0.7311],
         [ 0.8385,  1.7174,  0.0082],
         [ 0.8385,  1.7174,  0.0082],
         [ 0.8385,  1.7174,  0.0082]],

        [[ 0.6296,  0.0341,  0.5051],
         [-0.1341, -1.5869, -0.8380],
         [-0.1341, -1.5869, -0.8380],
         [-0.1341, -1.5869, -0.8380],
         [-0.1341, -1.5869, -0.8380]]])
```

注意使用高维索引是扩张了原始张量的shape大小的，这里为什么还是为3？

### 2.1 结果分解
经过测试，多维张量的索引需要保持shape一致？如ind.shape=[2,2,5]，两个索引张量shape是完全一致的。这里在第一个轴和第三个轴的取值是同步的！遍历分解如下：

```python
>>> pd_scores[0, :, [1,3,0,0,0]]
tensor([[-1.0320,  0.2565,  0.8385,  0.8385,  0.8385],
        [-0.5995,  0.7073,  1.7174,  1.7174,  1.7174],
        [ 0.0642, -0.7311,  0.0082,  0.0082,  0.0082]])
>>> pd_scores[0, :, [1,3,0,0,0]].T
tensor([[-1.0320, -0.5995,  0.0642],
        [ 0.2565,  0.7073, -0.7311],
        [ 0.8385,  1.7174,  0.0082],
        [ 0.8385,  1.7174,  0.0082],
        [ 0.8385,  1.7174,  0.0082]])
>>> pd_scores[1, :, [1,0,0,0,0]].T
tensor([[ 0.6296,  0.0341,  0.5051],
        [-0.1341, -1.5869, -0.8380],
        [-0.1341, -1.5869, -0.8380],
        [-0.1341, -1.5869, -0.8380],
        [-0.1341, -1.5869, -0.8380]])
```

### 2.2 索引说明

索引张量：

```python
>>> ind
tensor([[[0, 0, 0, 0, 0],    # 标识批量维度的索引
         [1, 1, 1, 1, 1]],

        [[1, 3, 0, 0, 0],    # 标识标注类别id的索引
         [1, 0, 0, 0, 0]]])
```

多轴高维张量索引：

```python
# `[ind[0], : ind[1]]`等价于:
# [
#     [[0, :, 1], [0, :, 3], [0, :, 0], [0, :, 0], [0, :, 0]]
#     [[1, :, 1], [1, :, 0], [1, :, 0], [1, :, 0], [1, :, 0]]
# ]
# 合并后的shape=[2,5,3]

```

### 2.3 多轴的索引张量shape必须可以广播

```python
>>> ind_ = torch.arange(end=3).view(-1, 1).repeat(1, 5)
>>> pd_scores[ind_, :, ind[1]]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: shape mismatch: indexing tensors could not be broadcast together with shapes [3, 5], [2, 5]
```

```python
>>> ind_ = torch.arange(end=1).view(-1, 1).repeat(1, 5)
>>> pd_scores[ind_, :, ind[1]]
tensor([[[-1.0320, -0.5995,  0.0642],
         [ 0.2565,  0.7073, -0.7311],
         [ 0.8385,  1.7174,  0.0082],
         [ 0.8385,  1.7174,  0.0082],
         [ 0.8385,  1.7174,  0.0082]],

        [[-1.0320, -0.5995,  0.0642],
         [ 0.8385,  1.7174,  0.0082],
         [ 0.8385,  1.7174,  0.0082],
         [ 0.8385,  1.7174,  0.0082],
         [ 0.8385,  1.7174,  0.0082]]])
# >>> ind_
# tensor([[0, 0, 0, 0, 0]])
# 被广播为：
# >>> ind_
# tensor([[0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0]])
```