# dl-project-colorization
2019 deep learning final project

### 模型
Colorizer.py提供了两种wrapper，分别用于unet上色和gan上色，可以在Networks中定义不同的具体网络用于初始化Colorizer wrapper

#### unet colorizer
colorizer = Colorizer(model=unet(1, 2))

#### gan colorizer
colorizer = GANColorizer(discriminator=discriminate_network(input_channel=2), generator=unet(1, 2))

#### 网络结构
[unet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

#### 训练
##### fit_generator
Colorizer wrapper使用Keras提供的fit_generator可以很方便地在每个epoch中进行batch训练。

其中generator参数为生成batch-manner数据的迭代生成器，epochs参数为epoch的次数，steps_per_epoch是每个epoch会训练多少个batch，其值为int(np.ceil(train_generator.size / batch_size))，这里取ceil而不是floor可以让数据集大小不能恰好被batch_size整除的时候也可以返回迭代数据，此时最后一个batch的样本数量会小于batch_size.

##### gan train_on_batch
由于GAN训练分为两步（第一步训练discriminator，第二步冻结discriminator参数训练gan），所以需要手动用for训练遍历data generator再调用GANColorizer中的train_op
```
for epoch_cnt in range(n_epoch):
        for batch_cnt, (x, y) in enumerate(tqdm(train_generator)):
            gan_loss, d_loss = colorizer.train_op(x, y)

```

### 数据
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
 
### utils
CONFIG.py主要定义了数据集的路径
visualize.py提供了写log的方法，可以在tensorboard中查看训练loss

### Prediction
对测试数据集进行测试的脚本为predict.py，test_dir为测试数据集（原始图片，非黑白图片！）的目录路径, res_dir为结果保存路径。

predict.py是一个简单的可以对单张图片数据进行predict并保存的脚本；用.ipynb可以比较好进行predict效果可视化展示。

