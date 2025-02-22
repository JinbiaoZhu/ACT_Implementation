# 手撕 ACT 算法并在 Bigym 上面训练/验证/测试

## 1. Why

### Q: 为什么要手撕这个代码？

我确实知道有现成的开源代码或项目完成了对 ACT 的复现，但是考虑自己所在实验室目前的现状，手撕一份简单一点的、可以上手单步调试的 ACT 代码是有益的：可以大致了解模仿学习模型结构、数据集规模和训练体量；了解模仿学习存在哪些改进空间；以及思考扩展到 VLA 的可能性。

### Q: 为什么选择 Bigym ？

这个仿真环境建模了人形机器人 H1 和一些交互任务，且仿真环境也提供了很多人类采集的数据（尽管目前发现只有 4 个任务具有 pixel-observation 的，但是还是很有用的）。Bigym 论文中对 ACT 做了测评，可以通过论文的数值进行对标自己的代码。同时这个 Bigym 也被 2024 年的 CoRL 会议接受了。

## 2. 参考资料

 - ACT 原始论文，在论文正文和附录里面讲的很清楚了，[链接](https://arxiv.org/pdf/2304.13705#page=8.86)。
 - Bigym 原始论文，[arxiv链接](https://arxiv.org/abs/2407.07788)，[OpenReview 情况](https://openreview.net/forum?id=EM0wndCeoD)。

## 3. 安装

```commandline
conda create -n rl310 python=3.10
```

来到主目录中，克隆项目并进入项目内安装依赖包。

```commandline
cd ~
```
```commandline
git clone https://github.com/JinbiaoZhu/ACT_Bigym.git
```
```commandline
cd ACT_Bigym
```
```commandline
pip install -r requirements.txt
```
安装完成后，需要在 *项目主目录* 中创建两个文件夹：
```commandline
mkdir ./ckpts
mkdir ./videos
```
创建完毕后，就可以跑代码了。
```commandline
python train.py
```
关于实验、模型和算法超参数等都可以在 ```./configs/ACT_config.py``` 中进行修改。

> 注意：本项目使用 Wandb 作为训练过程记录器，常规 Wandb 使用需要联网，且需要密钥，请事先准备。

# 4. 一些细节

 1. 数据集用类似 RL 的 replay buffer 收集的，但是实际项目或者真机实验应该换种方式。
 2. 在 ```simulation_test.py``` 的 ```test_in_simulation``` 函数中，我观察到 ```env.reset()``` 获得的本体数据和图像观测很奇怪，都是 0 ？所以我加了一步 random action 来获得更密集的信息。当然，这个 trick 是可以屏蔽的，直接注释掉即可。
 3. ```step_this_episode >= 1999``` 的 1999 是自己设置的，因为不成功的情况下机器人会一直奇怪的运动着，设置 1999 相当于提前截断了，比较省时间。
 4. 设置 ```[CLS]``` token, 变换成维度是 (1, 1, 1) 的张量, 并构建一个可学习的网络, 将维度 (1, 1, 1) 映射到 (1, 1, d_model) ，还有一种写法是直接将 ```[CLS]``` token 初始化成成维度是 (1, 1, d_model) 的张量, 但是这样写似乎不够直观...