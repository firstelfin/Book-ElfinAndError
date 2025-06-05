## docker error 1:  libnvidia-ml.so.1 : file exists: unknown

默认已经配置了镜像内的nvidia插件，宿主机不兼容这些插件导致报错。

- debug1: 尝试添加参数 `--runtime=nvidia` 进行指定
  - 若能成功，说明环境可以正常使用相关硬件了，若不能成功建议手动配置，查看debug2.
- debug2: 手动删除相关文件
  - 删除 libnvidia-ml.so.1 和 libcuda-ml.so.1，并保存为新的镜像，再启动.
  - 参考：https://blog.51cto.com/u_15642578/6178468
