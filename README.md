# dl-project-colorization
2019 deep learning final project

### Generate data in batch-manner
使用Dataset.py中的CifarGenerator来迭代生成数据，初始化参数img_dir为训练/测试图片所在的目录路径；
注意要通过is_training参数区别训练和测试（默认值为True），训练阶段会返回X和Y，测试阶段会返回X和对应图片的路径
```
generator = CifarGenerator(img_dir='./train_samples', batch_size=16, is_training=False)
#     if generator.is_training:
#         for x, y in generator:
#             print(x.shape)      # (batch_size, 32, 32, 1)
#             print(y.shape)      # (batch_size, 32, 32, 2)
#     else:
#         for x, f_names in generator:
#             print(x.shape)      # (batch_size, 32, 32, 1)
```

#### LAB编码
对图片进行编码不是用RGB color space而是用的LAB color space. LAB第一个通道其实就是灰度图，使用LAB通道进行网络训练时，输入的是lightness channel，即tensor shape为(batch_size, width, height, 1)；输出为剩下的两个channel，即tensor shape为(batch_size, width, height, 2)

The Lab color space describes mathematically all perceivable colors in the three dimensions L for lightness and a and b for the color components green–red and blue–yellow.

关于使用lab color space进行预处理的资料可以参考[about LAB color space 1](https://www.kaggle.com/preslavrachev/wip-photo-colorization-using-keras)中预处理的部分以及[about LAB color space 2](https://fairyonice.github.io/Color-space-defenitions-in-python-RGB-and-LAB.html)

#### 数据集
使用的是cifar数据，训练和测试的时候使用的数据都是彩色的而不是黑白数据!!
 
### Training
Colorizer提供了一个wrapper可以进行训练和测试，通过改变初始化model参数可以采用不同的网络结构。

#### 网络结构
[unet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

#### fit_generator
使用Keras提供的fit_generator可以很方便地在每个epoch中进行batch训练。

其中generator参数为生成batch-manner数据的迭代生成器，epochs参数为epoch的次数，steps_per_epoch是每个epoch会训练多少个batch，其值为int(np.ceil(train_generator.size / batch_size))，这里取ceil而不是floor可以让数据集大小不能恰好被batch_size整除的时候也可以返回迭代数据，此时最后一个batch的样本数量会小于batch_size.


### Prediction
对测试数据集进行测试的脚本为predict_batch.py，test_dir为测试数据集（原始图片，非黑白图片！）的目录路径, res_dir为结果保存路径。

predict.py是一个简单的可以对单张图片数据进行predict并保存的脚本；用.ipynb可以比较好进行predict效果可视化展示。

