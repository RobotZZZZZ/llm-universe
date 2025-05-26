# ContrastiveLoss 公式修正说明

## 问题描述

在原文档中，ContrastiveLoss 的公式存在一些不准确的地方：

### 原公式（存在问题）：
$$L=\frac{1}{n}\sum_{i=1}^{n}\{(1-label_i)\frac{1}{2}(cos \, dis(\vec{sent\_A_i},\vec{sent\_B_i}))^2+(label_i)\frac{1}{2}[max(0,margin-cos \, dis(\vec{sent\_A_i},\vec{sent\_B_i}))]^2\}$$

### 问题分析：

1. **术语混淆**：文档中使用了"余弦距离"，但实际上应该明确区分：
   - **余弦相似度**：范围 [-1, 1]，值越大表示越相似
   - **余弦距离**：通常定义为 `1 - 余弦相似度`，范围 [0, 2]，值越小表示越相似

2. **margin 的含义描述错误**：
   - 原文说"margin为label值为0的句对样本中sentence_A, sentence_B的最小距离阈值"
   - 实际上 margin 是用于 label=1（相似）样本的距离阈值

## 修正后的公式

### 标准的 Contrastive Loss 公式：

$$L=\frac{1}{n}\sum_{i=1}^{n}\{(1-label_i)\frac{1}{2}(distance)^2+(label_i)\frac{1}{2}[max(0,margin-distance)]^2\}$$

其中：
- `n` 为批次大小
- `label_i = 0` 表示不相似，`label_i = 1` 表示相似
- `distance` 是两个向量之间的距离（通常使用余弦距离：`1 - cosine_similarity`）
- `margin` 是期望的相似样本之间的最大距离阈值

### 损失函数的工作原理：

1. **当 label = 0（不相似）时**：
   - 损失项：`(1-0) * 1/2 * distance^2 = 1/2 * distance^2`
   - 目标：最大化距离，所以距离越小损失越大

2. **当 label = 1（相似）时**：
   - 损失项：`1 * 1/2 * [max(0, margin - distance)]^2`
   - 目标：最小化距离，当距离大于 margin 时产生损失

### 使用余弦相似度的变体：

如果直接使用余弦相似度（而不是余弦距离），公式需要相应调整：

$$L=\frac{1}{n}\sum_{i=1}^{n}\{(1-label_i)\frac{1}{2}[max(0,cos\_sim-margin)]^2+(label_i)\frac{1}{2}(1-cos\_sim)^2\}$$

这样：
- 不相似的样本：当余弦相似度大于 margin 时产生损失
- 相似的样本：余弦相似度越小损失越大

## 实际实现中的注意事项

在 Sentence Transformers 库中，ContrastiveLoss 的实际实现可能会有所不同，建议：

1. 查看具体的源代码实现
2. 根据实际使用的距离度量（欧氏距离、余弦距离等）来理解公式
3. 通过实验验证损失函数的行为是否符合预期

## 总结

原文档中的公式在以下方面需要修正：
1. 明确区分余弦相似度和余弦距离
2. 正确描述 margin 参数的含义
3. 确保公式的数学逻辑与损失函数的目标一致 