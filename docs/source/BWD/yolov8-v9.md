# yolov8与v9的抽风记录

[TOC]

--------------------------------------------------------

## 1. Silence、find_unused_parameters

无参数的恒等映射模块，不同于nn的恒等映射。

```python
class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x
```

在这里训练中曾出现某些参数未参与损失计算，在DDP初始化设置 `self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)`，设置后，训练又提示说没有参数没有使用，设置为true会影响效率。调试过程中，我也设置了Silence模块梯度为False。

```shell
Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance.
```

最后发现是我自己实现的PGI在验证的时候没有使用辅助分支，导致以上报错和警告！那么问题来了，每个epoch后的验证，调用损失计算是干什么？

-----------------------------------------------------

## 2. super().__init__()

类的继承这块涉及重构属性与方法的问题。方法在子类重构，那么子类肯定使用的是子类重构的方法。在`__init__`方法里书写的属性代码段实际上也是对应的重构行为。在对V8改造过程中，多次继承，部分`super().__init__()`没有传参，导致实例化的时候，子类没有按照传的参数进行实例化，如父类有`self.no=80`, 子类继承了，但是这个80是cfg里的一个参数，如果子类调用`super().__init__()`没有传参，那么最终的实例对象还是使用的默认值。
