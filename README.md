# ForecastBasedGraph



## 主要思路

	1、实体机器人通过云脑机器人架构，在云端形成了克隆模型的虚拟机器人，每个虚拟机器人拥有实体机器人的传感器数据。

    2、信息融合：通过将多维度的传感器数据通过Auto Encoder进行降维和去噪，形成高浓缩的特征，使得下游其他机器学习任务更加精确。

    3、利用基于LSTM的seq2seq进行预测

整体架构如下：
![fig1](./picture/fig6.png "fig6")

输入：(t,100,5)
输出：(t,10,1)
其实这里值得改进的问题是，如何使得模型也是实时在线的update?预测精度和周期都有待提高。

## 实验结果



1、实时传感器数据如下图所示：



![fig1](./picture/fig1.png "fig1")



![fig2](./picture/fig2.png "fig2")



2、通过Auoto Encoder降低到一维之后如下图所示：



![fig4](./picture/fig4.png "fig4")



3、利用seq2seq进行预测：利用前100个点(1s)的数据预测往后10(0.1s)的数据值



![fig5](./picture/fig5.png "fig5")







