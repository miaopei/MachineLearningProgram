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

