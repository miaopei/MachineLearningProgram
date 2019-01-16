# caffe note

## caffe 模型学习

```json
layer {
    name: "caffe_model"
    type: "Data"
    top: "data"  # 一般用bottom表示输入，top表示输出，多个top代表有多个输出
    top: "label"
    include {
    	phase: TRAIN  # 训练网络分为训练阶段和自测阶段，如果没写include则表示该层即在测试中，又在训练中
	}
	transform_param {
    	mean_file: "examples/caffe/mean.binaryproto" # 用一个配置文件进行均值操作
        transform_param {
        	scale: 0.00398625  # 1 / 255（归一化）
        	mirror: 1 # 1表示开启镜像，0表示关闭，也可用true和false来表示（数据增强操作）
        	# 剪裁一个227*227的图块，在训练阶段随机剪裁，在测试阶段从中间裁剪
        	crop_size: 227
    	}
	}
	data_param {
    	source: "examples/caffe/caffe_train_lmdb" # 数据库来源
        batch_size: 64 # 每次批处理的个数
        backend: LMDB # 选用数据的名称
	}
}

### 使用LMDB源
layer {
    name: "mnist"
    type: "Data"
    top: "data"
    top: "label"
    include {
    	phase: TRAIN
	}
	transform_param {
    	scale: 0.00398625
	}
	data_param {
    	source: "examples/caffe/mnist_train_lmdb"
        batch_size: 64 
        backend: LMDB 
	}
}

### 使用HDF5数据源
layer {
    name: "data"
    type: "HDF5Data"
    top: "data"
    top: "label"
    hdf5_data_param {
    	source: "examples/hdf5_classification/data/train.txt"
    	batch_size: 32
	}
}

### 数据直接来源于图片
# /path/to/images/img3423.jpg 2
# /path/to/images/img3424.jpg 13
# /path/to/images/img3425.jpg 8
layer {
    name: "data"
    type: "ImageData" # 类型
    top: "data"
    top: "label"
    transform_param {
    	mirror: false
    	crop_size: 227
    	mean_file: "data/caffe/imagenet_mean.binaryproto"
	}
	image_data_param {
    	source: "examples/_tmp/file_list.txt"
        batch_size: 64
        new_height: 256 # 如果设置就对图片进行resize操作
        new_width: 256
	}
}
```

```json
# 卷积层
layer {
    name: "conv1"
    type: "Convolution"
    bottom: "data"
    top: "conv1"
    param {
    	lr_mult: 1 # lr_mult: 学习率系数，最终的学习率是这个数乘以 solver.prototxt 配置文件中的 base_lr。如果有两个 lr_mult，则第一个表示权值的学习率，第二个表示偏置项的学习率。一般偏置项的学习率是权值学习率的两倍。如果lr_mult 设置为0，说明这个层权值或者偏置项不做更新。
	}
	param {
    	lr_mult: 2
	}
	convolution_param {
    	num_output: 20 # 卷积核（filter）的个数
        kernel_size: 5 # 卷积核的大小
        stride: 1 # 卷积核的步长，默认为1
        pad: 0 # 扩充边缘，默认为0，不扩充
        weight_filler {
        	type: "xavier" # 权值初始化，默认为“constant”，值为全0，很多时候我们用“xavier”算法来进行初始化，也可以设置为“gaussian”
    	}
		bias_filler {
    		type: "constant" # 偏置项的初始化。一般设置为“constant”，值全为0
		}
	}
}

# 输入： n*c0*w0*h0
# 输出： n*c1*w1*h1
# 其中，c1就是参数中的 num_output，
# 生成的特征图个数
#  w1 = (w0 - kernel_size + 2*pad) / stride + 1
#  h1 = (h0 - kernel_size + 2*pad) / stride + 1
```

```json
# 池化层
layer {
    name: "pool1"
    type: "Pooling"
    bottom: "conv1"
    top: "pool1"
    pooling_param {
    	pool: MAX # 池化方法，默认为 MAX。目前可用的方法有：MAX，AVE
    	kernel_size: 3 # 池化的核大小
    	stride: 2 # 池化的步长，默认为1。一般我们设置为2，即不重叠。
	}
}
# pooling层的运算方法基本是和卷积层是一样的。
```

```json
# 激活函数
# 在激活层中，对输入数据进行激活操作，是逐元素进行运算的，在运算过程中，没有改变数据的大小，即输入和输出的数据大小是相等的。

### sigmoid

layer {
    name: "test"
    bottom: "conv"
    top: "test"
    type: "sigmoid"
}

# ReLU 是目前使用最多的激活函数，主要因为其收敛更快，并且能保持同样的效果。标准的 ReLU 函数为 max(x,0)，当 x>0时，输出x；当x<=0时，输出0
# f(x) = max(x,0)

layer {
    name "relu1"
    type: "ReLU"
    bottom: "pool1"
    top: "pool1"
}
```

```json
# 全连接层
# 输出的是一个简单向量，参数跟卷积层一样
layer {
    name: "ip1"
    type: "InnerProduct"
    bottom: "pool2"
    top: "ip1"
    param {
    	lr_mult: 1
	}
	param {
    	lr_mult: 2
	}
	inner_product_param {
    	num_output: 500
        weight_filler {
        	type: "xavier"
    	}
		bias_filler {
    		type: "connstant"
		}
	}
}

# 测试的时候输入准确率
layer {
    name: "accuracy"
    type: "Accuracy"
    bottom: "ip2"
    bottom: "label"
    top: "accuracy"
    include {
    	phase: TEST
	}
}
```

```json
# softmax-loss layer: 输出loss值
layer {
    name: "loss"
    type: "SoftmaxWithLoss"
    bottom: "ip1"
    bottom: "label"
    top: "loss"
}

# softmax layer: 输出似然值
layer {
    bottom: "cls3_fc"
    top: "prob"
    name: "prob"
    type: "Softmax"
}
```

```json
# reshape 层
# 在不改变数据的情况下，改变输入的维度

layer {
    name: "reshape"
    type: "Reshape"
    bottom: "input"
    top: "output"
    reshape_param {
    	shape {
    		dim: 0 	# copy the dimension from below
    		dim: 2
    		dim: 3
    		dim: -1	# infer it from the other dimensions
		}
	}
}

# 有一个可选的参数组shape，用于指定blob数据的各维的值（blob是一个四维的数据：n*c*w*h）。
# dim:0  	表示维度不变，即输入和输出是相同的维度
# dim:2  	或 dim:3 将原来的维度变成 2 或 3
# dim:-1 	表示由系统自动计算维度。数据的总量不变，系统会根据blob数据的其他三维来自动计算当前维的维度值。
# 假设原数据为：32*3*28*28，表示32张3通道的28*28的彩色图片
	shape{
    	dim:0
        dim:0
        dim:14
        dim:-1
	}
# 输出数据为：32*3*14*56

# Dropout 是一个防止过拟合的层
# 只需要设置一个dropout_ratio就可以了。
layer {
    name: "drop7"
    type: "Dropout"
    bottom: "fc7-conv"
    top: "fc7-conv"
    dropout_param {
    	dropout_ratio: 0.5
	}
}
```

```json
# solver
####参数含义#############
# net: "examples/AAA/train_val.prototxt"    # 训练或者测试配置文件
# test_iter: 40   							# 完成一次测试需要的迭代次数
# test_interval: 475  						# 测试间隔
# base_lr: 0.01  							# 基础学习率
# lr_policy: "step"  						# 学习率变化规律
# gamma: 0.1  								# 学习率变化指数
# stepsize: 9500  							# 学习率变化频率
# display: 20  								# 屏幕显示间隔
# max_iter: 47500 							# 最大迭代次数
# momentum: 0.9 							# 动量
# weight_decay: 0.0005 						# 权重衰减
# snapshot: 5000 							# 保存模型间隔
# snapshot_prefix: "models/A1/caffenet_train" # 保存模型的前缀
# solver_mode: GPU 							# 是否使用GPU

# 往往loss function是非凸的，没有解析解，我们需要通过优化方法求解。
# caffe提供了六种优化算法求最优参数，在solver配置文件中，通过设置type类型选择。
# 	Stochastic Gradient Descent（type: "SGD"）
# 	AdaDelta（type: "AdaDelta"）
# 	Adaptive Gradient（type: "AdaGrad"）
# 	Adam（type: "Adam"）
#	Nesterov's Accelerated Gradient（type: "Nesterov"）
# 	RMSprop（type: "RMSProp"）

# lr_policy: "inv" # 学习率调整的策略
# 	- fixed: 	保持base_lr不变
# 	- step: 	如果设置为step，则还需要设置一个stepsize，返回 base_lr*gamma^(floor(iter/stepsize)), 其中iter表示当前的迭代次数
# 	- exp: 		返回 base_lr*gamma^iter，iter为当前迭代次数
# 	- inv: 		如果设置为inv，还需要设置一个power，返回base_lr*(1+gamma*iter)^(-power)
# 	- multistep: 如果设置multistep，则还需要设置一个stepvalue。这个参数和step很相似，step是均匀等间隔变化，而multistep则是根据stepvalue值变化
# 	- poly: 	学习率进行多项式误差，返回base_lr*(1-iter/max_iter)^(power)
# 	- sigmoid: 	学习率进行sigmoid衰减，返回base_lr*(1/(1+exp(-gamma*(iter-stepsize))))

net: "/home/tyd/caffe/examples/mnist/lenet_train_test.prototxt"
test_iter: 100	# 迭代了多少个测试样本？ batch*test_iter 假设有5000个测试样本，一次测试想跑遍这5000个则需要设置test_iter*batch=5000
test_interval: 500	# 测试间隔，也就是每训练500次，才进行一次测试。
base_lr: 0.01
lr_policy: "step"
momentum: 0.9
type: SGD
weight_decay: 0.0005
lr_policy: "inv"
gamma: 0.0001
power: 0.75
display: 100
max_iter: 20000
snapshot: 5000
snapshot_prefix: "models/A1/caffenet_train"
solver_mode: CPU
```

> [Caffe--solver.prototxt配置文件 参数设置及含义](https://www.cnblogs.com/Allen-rg/p/5795867.html)]



## 绘制网络

1、安装 graphViz

```shell
# sudo apt-get install graphViz
```

2、安装 pydot

```shell
# sudo apt-get install pydot
```

3、绘制网络

```shell
# sudo python /path/caffe/python/draw_net.py /path/lenet_train.prototxt /path/lenet.png --rankdir=BT
- 第一个参数：网络模型的prototxt文件
- 第二个参数：保存的图片路径及名字
- 第三个参数：--rankdir=x， x 有四种选项，分别是：
	- LR：从左到右
	- RL：从右到左
	- TB：从上到下
	- BT：从下到上
	- 默认为 LR
```











