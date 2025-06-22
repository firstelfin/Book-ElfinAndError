## 位置编码

作为非NLP出身的牛马，在江湖上经常听说苏建林，但是对于他的了解多限于科学空间，直到看RoPE论文才发现，这居然是苏神的贡献，不得不说对苏神的敬意又上升了一个维度。

### RoPE 旋转位置编码

[RoPE](https://arxiv.org/pdf/2104.09864)（2021v1～2023v5）

特点：
1. 随着相对距离的增加而衰减的token依赖性；
2. 为线性自注意力机制配备相对位置编码的能力；

创新性：
1. 提出旋转位置编码，通过将**上下文表示**与具有清晰理论解释的旋转矩阵相乘来编码相对位置（具有可解释的相对位置编码理论）；
2. RoPE随着相对距离的增加而衰减，这是符合自然语言的预期的，以前基于相对位置编码的方法与线性自注意力机制不兼容（没有体现衰减特性）；
3. RoFormer在所有长文本的benchmark上都有所提升，并提供了预训练模型。

---



#### 位置编码引入

设$\mathbb{S}_{N}=\left\{w_{i}\right\}_{i=1}^{N}$为长度为$N$的token ($w_{i}$：第i个输入)序列。$\mathbb{S}_{N}$对应的词嵌入是$\mathbb{E}_{N}=\left\{\boldsymbol{x}_{i}\right\}_{i=1}^{N}$。$x_{i}$是$d$维的词嵌入向量(不带位置编码)。自注意理机制首先将位置信息整合到词嵌入中，并将其转换为查询、键和值表示。

$$
\begin{align}
\boldsymbol{q}_m &= f_q(\boldsymbol{x}_m, m) \\
\boldsymbol{k}_n &= f_k(\boldsymbol{x}_n, n) \\
\boldsymbol{v}_n &= f_v(\boldsymbol{x}_n, n),
\end{align}
$$
query使用$m^{th}$位置合并，key、value使用$n^{th}$位置合并。然后，query和key-value用于计算注意力权重，而输出则计算为值表示的加权和。
$$
\begin{align}
a_{m,n} &= \frac{\exp\left(\frac{\boldsymbol{q}_{m}^\top \boldsymbol{k}_n}{\sqrt{d}}\right)}{\sum_{j=1}^N \exp\left(\frac{\boldsymbol{q}_m^\top \boldsymbol{k}_j}{\sqrt{d}}\right)} \\
\mathbf{o}_m &= \sum_{n=1}^N a_{m,n} \boldsymbol{v}_n
\end{align}
$$
现有的基于 transformer 的位置编码方法主要集中在选择合适的函数来形成方程$f$。



---



#### 绝对位置编码

绝对位置编码的$f$典型选择是：
$$
f_{t:t \in \{\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v}\}}(\boldsymbol{x}_i, i) := \mathbf{W}_{t:t \in \{\boldsymbol{q}, \boldsymbol{k}, \boldsymbol{v}\}}(\boldsymbol{x}_i + \boldsymbol{p}_i)
$$
其中$\boldsymbol{p}_i \in \mathbb{R}^{d}$是$d$维向量，取决于token $\boldsymbol{x}_i$的位置。18年～20年也有很多人使用可训练向量作为位置编码。AttentionIsAllYouNeed论文建议使用正弦函数生成位置编码$\boldsymbol{p}_i$：
$$
\begin{cases}
\boldsymbol{p}_{i,2t} &= \sin\left(\frac{k}{10000^{2t/d}}\right) \\
\boldsymbol{p}_{i,2t+1} &= \cos\left(\frac{k}{10000^{2t/d}}\right)
\end{cases}
$$
其中$\boldsymbol{p}_{i,2t}$是$d$维向量$\boldsymbol{p}_i$的$2t^{th}$元素（第$k$个元素，$k$从0开始计数）。在下一节中，我们从正弦函数的角度展示了我们提出的 RoPE 与这种直觉有关。然而，RoPE 不是直接将位置添加到上下文表示中，而是建议通过与正弦函数相乘来引入相对位置信息。

**频率缩放因子** ：

- 分母中的 $10000^{2t/d}$ 是一个频率缩放因子，用于控制不同维度的周期性变化频率。
- 10000 是一个经验值，用于确保不同(编码)维度的周期性变化具有不同的频率。
- $2t/d$ 是指数部分，决定了每个维度的频率。随着 *t* 的增加，指数部分逐渐增大，从而使得高频分量出现在高维部分。

**公式的核心思想**：

- **周期性** ：正弦和余弦函数具有周期性，因此位置编码也具有周期性。这种周期性使得模型能够捕捉到序列中位置的相对关系。
- **不同频率** ：通过调整  $10000^{2t/d}$  的指数部分，不同维度的正弦和余弦函数具有不同的频率。低维部分的频率较低，高维部分的频率较高，从而形成了丰富的周期模式。
- **位置信息** ：通过将位置索引 $k$ 代入公式，每个位置都会得到一个唯一的向量表示，从而为模型提供了位置信息。

##### 为神马使用正余弦函数函数交替？

- 核心原因：**保留信息的独立性和可学习性**

- 如果所有维度都只使用正弦函数（或只使用余弦函数），那么不同维度之间的表达会具有某种“对称性”或“重复性”，从而降低表示能力。

- 增强维度间的多样性

  - 正弦和余弦是正交函数，它们之间相差一个相位偏移 $sin(\boldsymbol{x})=cos(\boldsymbol{x}-\frac{\pi}{2})$
  - 两种函数可以让模型更容易区分不同维度的信号，增强每个维度的信息独立性

- **有利于**捕捉相对位置关系

  - 注意力模块中相对于绝对位置关系，更关注相对位置关系

  - sin 和 cos 的周期性使得，其他编码位置的位置编码可以由当前位置的正弦值和余弦值通过线性组合得到：这样模型可以通过线性操作轻松学习到相对位置信息。
    $$
    \begin{cases}
    sin(k+m)=a_{1} \cdot sin(k) + b_{1} \cdot cos(k)\\
    cos(k+m)=a_{2} \cdot sin(k) + b_{2} \cdot cos(k)
    \end{cases}
    $$

    > 很多人解读上面的公式实现了相对位置信息的捕捉，实际上是一种过度解读，作者使用正弦和余弦交替，在数学上将编码维度两两分组，确实能够实现上面的旋转变化，但是这只是一种引导设计，并不是显式建模，作者提供了原材料，让模型能够通过这种隐含的旋转关系(二维空间的线性变换)去学习相对位置关系，这也为RoPE提供的灵感。



---



#### 相对位置编码

2018年，论文《Self-attention with relative position representations》对方程$f$的设置如下：
$$
\begin{align}
f_q(\boldsymbol{x}_m) &:= \boldsymbol{W}_q \boldsymbol{x}_m \\
f_k(\boldsymbol{x}_n, n) &:= \boldsymbol{W}_k (\boldsymbol{x}_n + \widetilde{\boldsymbol{p}}_r^k) \\
f_v(\boldsymbol{x}_n, n) &:= \boldsymbol{W}_v (\boldsymbol{x}_n + \widetilde{\boldsymbol{p}}_r^v)
\end{align}
$$
其中$\widetilde{\boldsymbol{p}}_{r}^{k},\widetilde{\boldsymbol{p}}_{r}^{v}$可训练的相对位置嵌入。$r$是位置$m$和$n$之间的相对位置距离。他们通过假设相对位置信息在一定距离之外没有了，用来裁剪相对距离。

> 核心实现：
>
> 1. Key、Value使用了两组不同的可学习参数
> 2. 位置编码的选择是和$m,n$相关的，是一种可学习的相对位置编码

Xlnet论文提出将$\boldsymbol{q}_{m}^\top \boldsymbol{k}_n$分解为：
$$
\begin{align}
\boldsymbol{q}_{m}^\top \boldsymbol{k}_n &= ( \boldsymbol{W}_{q} (\boldsymbol{x}_{m} + \boldsymbol{p}_{m}))^{\top} \boldsymbol{W}_{k} (\boldsymbol{x}_{n} + \boldsymbol{p}_{n})\\
&= \boldsymbol{x}_{m}^{\top}\boldsymbol{W}_{q}^{\top}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + 
\boldsymbol{x}_{m}^{\top}\boldsymbol{W}_{q}^{\top}\boldsymbol{W}_{k}\boldsymbol{p}_{n} + 
\boldsymbol{p}_{m}^{\top}\boldsymbol{W}_{q}^{\top}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + 
\boldsymbol{p}_{m}^{\top}\boldsymbol{W}_{q}^{\top}\boldsymbol{W}_{k}\boldsymbol{p}_{n}
\end{align}
$$
然后，将$\boldsymbol{p}_{n}$替换为其正弦编码的相对位置编码$\widetilde{\boldsymbol{p}}_{m-n}$，第三项和第四项中的绝对位置编码$\boldsymbol{p}_{m}$替换为两个独立于查询位置的可训练向量$\mathbf{u},\mathbf{v}$。为了区分基于内容$\boldsymbol{x}_{n}$和位置$\boldsymbol{p}_{n}$的key对应权重，引入$\widetilde{\boldsymbol{W}}_{k}$表示对位置$\boldsymbol{p}_{n}$的对应权重。得到：
$$
\boldsymbol{q}_{m}^\top \boldsymbol{k}_n = \boldsymbol{x}_{m}^{\top}\boldsymbol{W}_{q}^{\top}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + 
\boldsymbol{x}_{m}^{\top}\boldsymbol{W}_{q}^{\top}\widetilde{\boldsymbol{W}}_{k}\widetilde{\boldsymbol{p}}_{m-n} + 
\mathbf{u}^{\top}\boldsymbol{W}_{q}^{\top}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + 
\mathbf{v}^{\top}\boldsymbol{W}_{q}^{\top}\widetilde{\boldsymbol{W}}_{k}\widetilde{\boldsymbol{p}}_{m-n}
$$
需要注意的是在Value对应的$f$函数中，直接舍弃了位置编码，即：$f_{v}(\boldsymbol{x}_{j}):=\boldsymbol{W}_{v}\boldsymbol{x}_{j}$。后续有很多论文遵循这种设置。

> 创新性：
>
> 1. 计算注意力时，将Key对应的位置编码直接换为可学习的相对位置编码
> 2. Key的调制权重，内容部分和位置编码部分使用不同的权重

Raffel等人再进一步，直接将上面的公式优化为：
$$
\boldsymbol{q}_{m}^\top \boldsymbol{k}_n = \boldsymbol{x}_{m}^{\top}\boldsymbol{W}_{q}^{\top}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + b_{i,j}
$$
其中是可训练的偏置，这种模式和CNN的偏置项非常相似（实际这里还是有考虑位置因素，CNN的bias没有）。因为研究发现这里绝对位置和单词之间几乎没有相关性，所以Raffel等人再次使用不同的投影矩阵对内容和位置分别建模：
$$
\boldsymbol{q}_{m}^\top \boldsymbol{k}_n = \boldsymbol{x}_{m}^{\top}\boldsymbol{W}_{q}^{\top}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + 
\boldsymbol{p}_{m}^{\top}\mathbf{U}_{q}^{\top}\mathbf{U}_{k}\boldsymbol{p}_{n} + 
b_{i,j}
$$
Deberta论文认为，两个token的相对位置可以由使用Xlnet分解公式中的中间两项来完全建模
$$
\boldsymbol{q}_{m}^\top \boldsymbol{k}_n = \boldsymbol{x}_{m}^{\top}\boldsymbol{W}_{q}^{\top}\boldsymbol{W}_{k}\boldsymbol{x}_{n} + 
\boldsymbol{x}_{m}^{\top}\boldsymbol{W}_{q}^{\top}\widetilde{\boldsymbol{W}}_{k}\widetilde{\boldsymbol{p}}_{m-n} + 
\widetilde{\boldsymbol{p}}_{m-n}^{\top}\boldsymbol{W}_{q}^{\top}\boldsymbol{W}_{k}\boldsymbol{x}_{n}
$$
后续的研究，有人认为这种结构更加有效。RoPE也将基于此进行推导。



---



#### RoPE旋转位置编码的推导历程

##### 基础公式

基于 Transformer 的语言建模通常利用单个token的位置信息注入到自注意机制。从上面注意力公式可以看出，$\boldsymbol{q}_{m}^\top \boldsymbol{k}_n$通常支持在不同位置的token之间进行知识传递。为了嵌入相对位置信息，要求注意力的由$\boldsymbol{q}_{m}$和$\boldsymbol{k}_{n}$的内积函数值表示，这个函数应该只由token的内容编码$\boldsymbol{x}_{m}$、$\boldsymbol{x}_{n}$及其相对位置$m-n$作为输入变量。即希望内积只以相对位置编码：
$$
\langle f_q(\boldsymbol{x}_m, m), f_k(\boldsymbol{x}_n, n) \rangle = g(\boldsymbol{x}_m, \boldsymbol{x}_n, m - n)
$$
最终目标是找到一种等效的编码机制求解函数$f_q(\boldsymbol{x}_m, m), f_k(\boldsymbol{x}_n, n)$来符合上述关系。



---



##### 旋转位置嵌入

利用 2D 平面上向量的几何性质及其复数形式来证明上面的要求的解可以为：
$$
\begin{align}
f_q(\boldsymbol{x}_m, m) &= (\boldsymbol{W}_q \boldsymbol{x}_m)e^{im\theta} \\
f_k(\boldsymbol{x}_n, n) &= (\boldsymbol{W}_k \boldsymbol{x}_n)e^{in\theta} \\
g(\boldsymbol{x}_m, \boldsymbol{x}_n, m-n) &= \operatorname{Re}\left[(\boldsymbol{W}_q \boldsymbol{x}_m)(\boldsymbol{W}_k \boldsymbol{x}_n)^* e^{i(m-n)\theta}\right]
\end{align}
$$
其中$\operatorname{Re} \left[ \cdot \right]$是复数的实部，$(\boldsymbol{W}_k \boldsymbol{x}_n)^*$表示$(\boldsymbol{W}_k \boldsymbol{x}_n)$的共轭复数。$\theta \in \mathbb{R}$是预设的非零常数。我们可以进一步写$f_{\{q,k\}}$为乘法矩阵：
$$
\begin{align}
f_{\{q,k\}}(\boldsymbol{x}_m, m) =
\begin{pmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{pmatrix}
\begin{pmatrix}
W^{(11)}_{\{q,k\}} & W^{(12)}_{\{q,k\}} \\
W^{(21)}_{\{q,k\}} & W^{(22)}_{\{q,k\}}
\end{pmatrix}
\begin{pmatrix}
x_m^{(1)} \\
x_m^{(2)}
\end{pmatrix}
\end{align}
$$
上式右边最后一项是的2D坐标表示。同样，$g$可以看作是一个矩阵，因此可以在 2D 情况下解决【基础公式】章节的需求。具体来说，内积函数中显式引入相对位置关系，只需要将内容嵌入向量旋转由绝对位置索引的角度即可，因为$f_{q}$和$f_{k}$进行内积时，旋转角度实际就是一个相对角度，这样就将相对位置关系显式引入建模过程中了。

为了推广到高维的嵌入空间，苏神将token编码维度划分为$d/2$个子空间，并利用内积是线性的优点，对于高维的$f_{\{q,k\}}$有：
$$
f_{\{q,k\}}(\boldsymbol{x}_m, m) = \boldsymbol{R}_{\Theta,m}^d \boldsymbol{W}_{\{q,k\}} \boldsymbol{x}_m
$$
其中，
$$
\boldsymbol{R}_{\Theta,m}^d =
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix}
$$
旋转矩阵的参数是预先定义的，$\Theta = \left\{ \theta_i = 10000^{-2(i-1)/d}, i \in [1, 2, \dots, d/2 ] \right\}$。因此，RoPE的自注意力为：
$$
\boldsymbol{q}_m^\top \boldsymbol{k}_n = (\boldsymbol{R}_{\Theta,m}^d \boldsymbol{W}_q \boldsymbol{x}_m)^\top (\boldsymbol{R}_{\Theta,n}^d \boldsymbol{W}_k \boldsymbol{x}_n) = \boldsymbol{x}_m^\top \boldsymbol{W}_q \boldsymbol{R}_{\Theta,n-m}^d \boldsymbol{W}_k \boldsymbol{x}_n
$$
注意：这里$\boldsymbol{R}_{\Theta,n-m}^d = (\boldsymbol{R}_{\Theta,m}^d)^{\top}\boldsymbol{R}_{\Theta,n}^d$，RoPE在注意力的内积计算过程中，通过两个旋转矩阵的乘积实现了相对位置编码信息的引入，而且这里是显示的引入。$\boldsymbol{R}_{\Theta}^d$是一个正交矩阵，保证了编码位置信息过程中的稳定性，由于其稀疏性，上面的数学公式直接使用矩阵乘法的效率是不高的。



---



##### RoPE的特性

**相对位置较大时衰减**

当相对位置增加时，内积会衰减。这个属性与一对相对距离较长的标记应该具有较少连接的直觉相吻合。

**线性注意力的RoPE**

自注意力机制可以写为：
$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_m = \frac{\sum_{n=1}^N \text{sim}(\boldsymbol{q}_m, \boldsymbol{k}_n) \boldsymbol{v}_n}{\sum_{n=1}^N \text{sim}(\boldsymbol{q}_m, \boldsymbol{k}_n)}
$$
原来的自注意力选择了$\text{sim}(\boldsymbol{q}_m, \boldsymbol{k}_n)=\exp (\boldsymbol{q}_m^\top \boldsymbol{k}_n / \sqrt{d})$。原始 self-attention 应该计算每对 tokens 的 query 和 key 的内积，它具有$\mathcal{O}(N^2)$的复杂度(这也是制约Transformer发展的关键因素)。“Transformers are rnns”论文改写了上面的公式：
$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_m = \frac{\sum_{n=1}^N \phi(\boldsymbol{q}_m)^{\top} \varphi( \boldsymbol{k}_n) \boldsymbol{v}_n}{\sum_{n=1}^N \phi(\boldsymbol{q}_m)^{\top} \varphi( \boldsymbol{k}_n)}
$$
其中$\phi(\cdot),\varphi(\cdot)$通常是非负函数。后续也有很多论文关于这两个函数有不同的设置。由于 RoPE 通过旋转注入位置信息，这保持了隐藏表示的规范不变，我们可以通过将旋转矩阵与非负函数的输出相乘，将 RoPE 与线性注意力相结合。
$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_m = \frac{\sum_{n=1}^N \left( \boldsymbol{R}_{\Theta,m}^d \phi(\boldsymbol{q}_m) \right)^\top \left( \boldsymbol{R}_{\Theta,n}^d \varphi(\boldsymbol{k}_n) \right) \boldsymbol{v}_n}{\sum_{n=1}^N \phi(\boldsymbol{q}_m)^\top \varphi(\boldsymbol{k}_n)}
$$
值得注意的是，我们保持分母不变以避免除以零的风险，并且分子中的求和可能包含负项。尽管上面公式中的$\boldsymbol{v}_{i}$权重没有严格地进行概率归一化，但我们仍然认为，计算可以对值(value)的重要性进行建模。



---



RoPE 3.4章节详细说明了在2D模式下的RoPE推导，强烈推荐查看原论文。本文是基于苏神的论文直接翻译的，翻译软件部分翻译是不对的，甚至和实际的意思完全相反，若有错漏的部分，以原始论文为主。



---



