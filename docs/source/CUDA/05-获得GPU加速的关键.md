<p>
    <center><h1>获得GPU加速的关键</h1></center>
	<br />
    <p name="top" id="top" align="center">
        <b>作者：</b><b><a href="https://www.cnblogs.com/dan-baishucaizi/">elfin</a></b>&nbsp;&nbsp;
        <b>资料来源：<a href="">CUDA编程与实践</a></b>
	</p>
</p>


---

[TOC]

---

​	前几章主要关注程序的正确性，没有强调程序的性能（执行速度）。在开发CUDA程序时往往要验证某些改变是否提高了程序的性能，这就需要对程序进行比较精确的计时。所以我们就从给主机核设备函数的计时讲起。

---

# 5.1 用CUDA事件计时

​	在c++中，有多种可以对一段代码进行计时的方法，包括使用GCC和MSVC都有的clock函数和头文件`<chrono>`对应的时间库、GCC中的`gettimeofday()`函数及MSVC中的`QueryPerformanceCounter()`和`QueryPerformanceFrequency()`函数等。CUDA提供了一种基于CUDA事件（CUDA event）的计时方式，可用来给一段CUDA代码(可能包含主机代码和设备代码)计时。为简单起见，我们这里仅介绍基于CUDA事件的计时方法。下面我们给出常用的计时方式：

```c++
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));
CHECK(cudaEventRecord(start));
cudaEventQuery(start);  // 此处不能用CHECK宏函数

需要计时的代码块

CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));
float elapsed_time;
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
printf("Time = %g ms .\n", elapsed_time);
CHECK(cudaEventDestroy(start));
CHECK(cudaEventDestroy(stop));
```

其中：

*   (1) 第1行定义了两个CUDA事件类型(cudaEvent_t)的变量start和stop，第2行和第3行用cudaEventCreate()函数初始化它们。
*   (2) 第4行将start传入cudaEventRecord()函数，在需要计时的代码块之前记录一个代表开始的事件。
*   (3) 第5行对处于TCC驱动模式的GPU来说可以省略，但对处于WDDM驱动模式的GPU来说必须保留。因为，在处于WDDM驱动模式的GPU中，一个CUDA流（CUDA stream）中的操作（如这里的cudaEventRecord()函数）并不是直接提交给GPU执行，而是先提交到一个软件队列，需要添加一条对该流的cudaEventQuery操作（或者cudaEventSynchronize）刷新队列，才能促使前面的操作在GPU执行。
*   (4) 第7行代表一个需要计时的代码块，可以是一段主机代码，也可以是一段设备代码，还可以是一段混合代码。
*   (5) 第9行将stop传入cudaEventRecord()函数，在需要计时的代码块之后记录一个代表结束的事件。
*   (6) 第10行的cudaEventSynchronize()函数让主机等待时间stop被记录完毕。
*   (7) 第11~13行调用cudaEventElapsedTime()函数计算start到stop的时间差并输出到屏幕(单位：ms)。
*   (8) 第15~16行调用cudaEventDestroy()函数销毁start和stop这两个CUDA事件。这是本笔记唯一使用事件的地方，故这里不对CUDA事件做进一步讨论。下面我们尝试对数组相加的代码块计时。

---

## 5.1.1 为C++程序计时

这里先考虑C++版本的程序。我们使用add1cpu.cu验证为C++程序计时，相对于第三章的add.cpp有如下的改动：

* (1) 即使该程序没有核函数，我们也将源文件的扩展名写为`.cu`，这样就不用包含一些CUDA头文件了。若还是使用`.cpp`，用nvcc 编译时需要明确地增加一些头文件的包含，用gcc编译时还要明确地链接一些CUDA库。

* (2) 从现在起，我们用调节编译的方式选择程序中所用浮点数的精度。在程序开头部分，添加如下几行代码：

  ```c++
  #ifdef USE_DP
      typedef double real;
      const real EPSILON = 1.0e-15;
  #else
      typedef float real;
      const real EPSILON = 1.0e-6f;
  #endif
  ```

  当宏USE_DP有定义时，程序的real代表double，否则代表float。该宏可以通过编译选项定义(具体见后吗的编译命令)。

* (3) 我们用CUDA时间对该程序中函数add()的调用进行了计时，而且重复了11次。我们忽略第一次的时间，因为第一次机器可能处于预热状态，测得的时间往往偏大。我们根据后10次测得时间的平均值标记add的执行用时。详情参考add1cpp.cu(我的实现)、add1cpp_.cu(樊老师的实现)。



**我们使用nvcc进行编译，需要注意以下几点：**

* (1) C++程序的性能显著地依赖于优化选项，我们将总是使用`-O3`选项。
* (2) 编译时使用`-DUSE_DP`的命令行参数，程序中的宏将有定义，从而使用双精度浮点数，否则使用单精度浮点数。

编译命令：

```shell
$ nvcc -O3 -arch=sm_86  add1cpu.cu -o add1float
$ ./add1float
Time = 171.325 ms .
No errors
$ nvcc -O3 -arch=sm_86  add1cpu_.cu -o add1float_
$ ./add1float_
Time = 220.226 ms.
Time = 87.8418 ms.
Time = 88.4715 ms.
Time = 89.0419 ms.
Time = 88.2954 ms.
Time = 86.6734 ms.
Time = 86.9325 ms.
Time = 86.8219 ms.
Time = 87.2632 ms.
Time = 87.2272 ms.
Time = 87.2445 ms.
Time = 87.5813 +- 0.7526 ms.
No errors
$ nvcc -O3 -arch=sm_86 -DUSE_DP  add1cpu.cu -o add1double
$ ./add1double
Time = 176.765 ms .
No errors
$ nvcc -O3 -arch=sm_86 -DUSE_DP  add1cpu_.cu -o add1double_
$ ./add1double_
Time = 399.767 ms.
Time = 175.862 ms.
Time = 175.972 ms.
Time = 176.373 ms.
Time = 175.953 ms.
Time = 176.814 ms.
Time = 176.315 ms.
Time = 175.655 ms.
Time = 176.055 ms.
Time = 176.876 ms.
Time = 175.824 ms.
Time = 176.17 +- 0.392806 ms.
No errors
```

这里add1cpu_.cu的双精度版本比单精度版本慢2倍，这和我们的预期是一样的，但是add1cpu.cu的双精度单精度却是一样的！这是为什么呢？

> TODO：执行时间一样问题查找？
>
> 问题解答，这是因为我自己写的代码中没有使用宏中的real修饰变量，数值还是用的double进行修饰，所以执行用时一样。可自行修改测试执行用时...

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 5.1.2 为CUDA程序计时

类似C++程序计时，我们在第四章的check1api.cu的基础上进行修改，用CUDA时间对其中的核函数add()进行计时，从而得到本章的add2gpu.cu程序。我们用如下命令进行编译：

```shell
$ nvcc -arch=sm_86 -DUSE_DP add2gpu.cu -o add2double
$ ./add2double 
Time = 7.56797 ms.
Time = 7.97798 ms.
Time = 7.53677 ms.
Time = 7.83462 ms.
Time = 7.50899 ms.
Time = 8.01997 ms.
Time = 7.13699 ms.
Time = 7.88416 ms.
Time = 7.13318 ms.
Time = 7.72618 ms.
Time = 7.13318 ms.
Time = 7.5892 +- 0.336505 ms.
No errors
$ nvcc -arch=sm_86 add2gpu.cu -o add2float
$ ./add2float
Time = 3.57258 ms.
Time = 3.57251 ms.
Time = 3.6248 ms.
Time = 3.57923 ms.
Time = 3.57069 ms.
Time = 3.58179 ms.
Time = 3.57171 ms.
Time = 3.61546 ms.
Time = 3.58067 ms.
Time = 4.18131 ms.
Time = 3.56659 ms.
Time = 3.64448 +- 0.179908 ms.
No errors
```

这里双精度的执行时间几乎是单精度的两倍，符合我们的预期。

​	这里我测试的显卡是RTX3060，显存带宽360GB/s。我们可以计算有效显存带宽，并与理论显存带宽进行比较。有效显存带宽的定义为GPU在单位时间内访问设备内存的字节。以RTX3060为例：
$$
\frac{3 \times 10^{8} \times 4 \text{B}}{3.64  \times 10^{-3}} = 326 \,\text{GB/s}
$$
可见有效显存要小于理论显存，这也说明了我们的add2gpu.cu是访存主导，浮点数运算所占比例可以忽略不计。(你可以这样理解，显存带宽有效值肯定比理论值小，我们就按照360去算访问显存占用的时间差不多在3.3ms，这样计算就只有0.34ms)。

除了访问显存是核函数执行的重要占时操作，我们想看一下数据从主机拷贝到设备上的用时。这里我们使用add3memcpy.cu进行测试。

```shell
$ nvcc -arch=sm_86 add3memcpy.cu -o add3float
$ ./add3float 
Time = 289.711 ms.
Time = 117.61 ms.
Time = 118.675 ms.
Time = 117.929 ms.
Time = 118.078 ms.
Time = 118.026 ms.
Time = 118.094 ms.
Time = 119.088 ms.
Time = 118.102 ms.
Time = 118.037 ms.
Time = 118.33 ms.
Time = 118.197 +- 0.392806 ms.
No errors
$ nvcc -arch=sm_86 -DUSE_DP add3memcpy.cu -o add3double
$ ./add3double 
Time = 570.68 ms.
Time = 232.299 ms.
Time = 232.803 ms.
Time = 232.926 ms.
Time = 234.929 ms.
Time = 235.058 ms.
Time = 233.646 ms.
Time = 233.025 ms.
Time = 232.505 ms.
Time = 231.835 ms.
Time = 231.913 ms.
Time = 233.094 +- 1.07165 ms.
No errors
```

根据以上结果我们不难看出，核函数的执行时间不到数据复制的4%。程序相对于C++程序不是性能提升而是性能降低。如果一个程序的计算任务仅仅是将来自主机端的两个数组相加，并要将结果传回主机端，使用GPU就不是一个明智的选择。那么什么样的任务使用GPU可以加速呢？这个我们需要通过nvprof工具进行分析。

**CUDA程序的性能剖析**

```shell
$ nvprof ./add3float
======== Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
                  Use NVIDIA Nsight Systems for GPU tracing and CPU sampling and NVIDIA Nsight Compute for GPU profiling.
                  Refer https://developer.nvidia.com/tools-overview for more details.
# 这里给出了工具不适配的警告，nvprof工具不支持计算能力8.0之上的，RTX3060计算能力应该是8.0
$ nsys nvprof ./add3float
CUDA API Statistics:

 Time(%)  Total Time (ns)  Num Calls    Average      Minimum      Maximum            Name        
 -------  ---------------  ---------  ------------  ----------  -----------  --------------------
    88.8    1,495,744,336         33  45,325,585.9  35,201,658  210,137,518  cudaMemcpy          
    10.9      183,403,542          3  61,134,514.0     412,713  182,565,956  cudaMalloc          
     0.2        4,039,907          3   1,346,635.7     494,647    1,997,116  cudaFree            
     0.0          301,560         11      27,414.5      26,690       29,114  cudaLaunchKernel    
     0.0           92,742         22       4,215.5       1,652        9,172  cudaEventRecord     
     0.0           79,943         22       3,633.8         421       13,632  cudaEventCreate     
     0.0           32,765         22       1,489.3         417       12,990  cudaEventDestroy    
     0.0           30,179         11       2,743.5       2,493        3,556  cudaEventSynchronize
     0.0           11,574         11       1,052.2         759        1,997  cudaEventQuery
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances    Average     Minimum    Maximum                       Name                    
 -------  ---------------  ---------  -----------  ---------  ---------  --------------------------------------------
   100.0       39,326,716         11  3,575,156.0  3,565,614  3,610,350  add(float const*, float const*, float*, int)



CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations    Average      Minimum      Maximum        Operation     
 -------  ---------------  ----------  ------------  ----------  -----------  ------------------
    63.9      927,914,267          22  42,177,921.2  41,430,370   42,973,257  [CUDA memcpy HtoD]
    36.1      523,959,305          11  47,632,664.1  31,440,315  205,787,528  [CUDA memcpy DtoH]

```

这里将所有操作的时间都列出来了，但是核函数和数据传输没有放一起比较，我们可以简单计算出，核函数执行时间只占主机拷贝数据到设备执行时间的4.24%。这个比例非常小，所以这种操作不适合使用GPU进行计算。



>   如果上面的操作下面的错误提示：
>
>   ```shell
>   $ nvprof ./add3float
>   Unable to profile application. Unified Memory profiling faild
>   # 可以尝试下面的命令
>   $ nvprof --unified-memory-profiling ./add3float
>   ```

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

# 5.2 影响GPU加速的几个关键因素

## 5.2.1 数据传输所占的比例

如上一节所述，如果数据传输占了执行时间的大头，那GPU可能比CPU还慢。GPU计算核心和设备内存间的传输峰值理论带宽要远高于GPU与CPU之间数据传输的带宽。以我的3060为例，显存带宽为$360 \, \text{GB/s}$，而GPU与CPU内存的PCIe3只有$16 \, \text{GB/s}$。虽然现在是PCIe4和PCIe5，但是带宽也只有$64 \, \text{GB/s}$左右。这都远小于显存带宽，所以要获得客观的GPU加速，就必须尽量缩减数据传输所花时间的比例。有时，即使计算在GPU中的速度并不高，也要在尽量在GPU中实现，**避免过多的PCIe传输**。这是CUDA编程中重要的原则。



## 5.2.2 算术强度

​	一个计算问题的**算术强度**指的是其中**算术操作的工作量与必要的内存操作的工作量之比**。

​	以3060为例，显存带宽为$360 \, \text{GB/s}$，理论寄存器带宽为：(3060的浮点性能为12.74TF)
$$
\frac{4\text{B} \times 4(每个FMA的操作数量)}{2(每个FMA浮点数操作次数)} \times 12.74 \times 10^{12} \text{/s} = 102 \, \text{TB/s}
$$
RTX2070的寄存器带宽是$52 \, \text{TB/s}$，3060几乎是翻了一倍，着实不错。但是前面的加法是计算强度小的案例，所以执行时间，3060并没有比2070有优势，理论上如果计算占比更高，那么3060的性能更好。这里的FMA是指fused multiply-add指令，及涉及4个操作数和两个浮点数操作的运算$d=a\times b + c$。说明这里GPU的浮点数计算比GPU的数据存取要快290多倍。如果是双精度浮点数，也要快几倍。注意RTX显卡不是算力卡，所以双精度计算性能很差，如果有双精度浮点数的计算需求，可以买Tesla这种卡。

为了增加计算强度，我们进行如下的修改。

**arithmetic1cpu.cu中的arithmetic函数**

```c++
void arithmetic(real *x, const real x0, const int N)
{
    for (int n = 0; n  < N; ++n)
    {
        real x_temp = x[n];
        while (sqrt(x_temp) < x0)
        {
            ++x_temp;
        }
        x[n] = x_temp;
    }
}
```

**arithmetic2gpu.cu中的arithmetic函数**

```c++
void __global__ arithmetic(real *d_x, const real x0, const int N)
{
    const int n = blockDim.x  * blockIdx.x + threadIdx.x;
    if (n>=N) return;
    real x_temp = d_x[n];
    while (sqrt(x_temp) < x0)
    {
        ++x_temp;
    }
    d_x[n] = x_temp;
}
```

​	这里有个real类型的常亮$x0$，控制着加法计算的次数(相当于之前的外层循环，增加计算强度)。要注意的是循环条件中有sqrt()函数，源书籍中常量为100，那就会循环10000次，我的设备c++程序耗不起，所以将常量设为10，只循环100次。

**编译执行结果：**

```shell
$ nvcc -O3 -arch=sm_80 arithmetic1cpu.cu -o arithmetic1float
$ ./arithmetic1float 
Time = 42510.5 ms .
```

在CUDA版本中，我们加入了如下的代码块控制N从命令行获取数值：

```c++
if (argc != 2) 
    {
        printf("usage: %s N\n", argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);
```

所以编译后，执行的时候要添加命令行参数N

```shell
$ nvcc -arch=sm_80 arithmetic1cpu.cu -o arithmetic1float
$ nvcc -O3 -arch=sm_80 arithmetic1cpu.cu -o arithmetic1float
$ nvcc -O3 -USE_DP -arch=sm_80 arithmetic1cpu.cu -o arithmetic1double
$ nvcc -O3 -arch=sm_80 arithmetic2gpu.cu -o arithmetic2float
$ nvcc -O3 -USE_DP -arch=sm_80 arithmetic2gpu.cu -o arithmetic2double
$ ./arithmetic1float 10000
Time = 5.39046 ms .
$ ./arithmetic2float 10000
Time = 0.023936 ms.
$ ./arithmetic1double 10000
Time = 8.14582 ms .
$ ./arithmetic2double 10000
Time = 0.142912 ms.
```

测试中：当数组长度为$10^{4}$时，主机函数的执行时间是5.39046 ms（单精度）和8.14582 ms（双精度）；CUDA程序的执行时间是0.023936 ms（单精度）和0.142912 ms（双精度）。所以单精度浮点数和双精度浮点数的加速比为：

单精度：
$$
\frac{5.39046}{0.023936}=225
$$
双精度：
$$
\frac{8.14582}{0.142912}=57
$$
我们可以看到算数强度的提高，加速比也极大地提高了。当N为$10^{6}$时，单精度、双精度的加速比分别为1681、58。可以发现单精度版本提升计算强度，加速效果比双精度好太多，注意我使用的是RTX3060(这种个人游戏卡对双精度计算几乎是啥优化)。在RTX3060显卡中核函数双精度与单精度的执行时间比是44倍，还是比较接近理论的32倍，所以侧面印证了这个核函数是计算主导的。Tesla V100双精度是单精度的2倍，其他方面并没有突出优势，所以对于AI计算，GeForce系列显卡性价比还是要高一些。

---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 5.2.3 并行规模

​	影响CUDA程序性能的另一个因素是并行规模。并行规模可以用GPU中总的线程数目来衡量。从硬件上说，一个GPU由多个流处理器(SM)构成，而每个SM中有若干CUDA核心。每个SM是相对独立的。从开普勒到伏特架构，一个SM最多能驻留2048个线程；现在我们使用的图灵、安培架构是1024.一块GPU中一般有几个到几十个SM单元。所以一个显卡一共可以驻留几万到几十万个线程，如果一个核函数定义的线程数目远小于这个数，就很难获得很高的加速比。

​	为了测试RTX3060，这里我们将N从$10^{3}$逐渐增到到$10^{8}$，观察核函数的执行时间和加速比：



---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

## 5.2.4 总结

一个CUDA程序能获得高性能必须满足以下几点：

* (1) 数据传输比例较小；
* (2) 核函数的算术强度较高；
* (3) 核函数中定义的线程数目较多。

所以在写CUDA程序时，应该做到以下几点：

* (1) 减少设备与主机之间的数据传输；
* (2) 提高核函数的算术强度；
* (3) 增大核函数的并行规模。



---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>












---

<p align="right">
    <b><a href="#top">Top</a></b>
	&nbsp;<b>---</b>&nbsp;
	<b><a href="#bottom">Bottom</a></b>
</p>

<p name="bottom" id="bottom">
    <b>完！</b>
</p>