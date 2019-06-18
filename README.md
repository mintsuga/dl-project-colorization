# dl-project-colorization
2019 deep learning final project

### 模型
models目录下提供了两种模型，一个是直接使用unet不用GAN，另一个是使用conditional GAN，conditional GAN的generator也使用了unet的结构，discriminator接受一个condition图像来判断生成图像是fake还是real，discriminatory是一个四层的卷积网络（without pooling），最后输出一个标量的validation值，表示生成图像fake/real的程度。

#### 训练

两种模型的训练均在train.py中，分别为train_gan(sess)和train_unet(sess)；

use_logs参数用来控制tensorboard记录loss，unet模型使用Keras的fit_generator接口，tensorboard每完成一个epoch通过callback记录一次；而cgan则是每个step通过tf.summary的接口手动写入；

通过运行tensorboard --logdir='./logs'可以查看loss

![unet loss](./results/unet_loss.png) 

![gan loss](./results/gan_loss.png) 

##### fit_generator
Colorizer wrapper使用Keras提供的fit_generator可以很方便地在每个epoch中进行batch训练。

其中generator参数为生成batch-manner数据的迭代生成器，epochs参数为epoch的次数，steps_per_epoch是每个epoch会训练多少个batch，其值为int(np.ceil(train_generator.size / batch_size))，这里取ceil而不是floor可以让数据集大小不能恰好被batch_size整除的时候也可以返回迭代数据，此时最后一个batch的样本数量会小于batch_size.

##### cgan train_on_batch
conditional GAN中的images_A和images_B分别是原始彩色图和灰度图，其训练步骤如下：

1. 第一步通过generator.predict()生成fake的彩色图。
2. 以灰度图为conditional images，分别以原始彩色图和生成彩色图训练discriminator，

```
d_loss_real = colorizer.discriminator.train_on_batch([images_A, images_B], valid)
d_loss_fake = colorizer.discriminator.train_on_batch([fake_A, images_B], fake)

```

### 数据
使用Dataset.py中的CifarGenerator来迭代生成数据，初始化参数img_dir为训练/测试图片所在的目录路径；可以通过设置color_space参数决定使用LAB编码还是RGB编码；
对于每次迭代返回的X和Y，LAB编码会分别得到shape为(batch_size, 32, 32, 1), (batch_size, 32, 32, 2)的数据，x表示lightness通道，y表示剩下的两个通道；
而RGB得到的都是(batch_size, 32, 32, 3)的，但是y是灰度图，x是原始彩色图。

实际实验时unet模型使用的是LAB编码，而cgan则采用了RGB编码

#### 关于LAB编码
对图片进行编码不是用RGB color space而是用的LAB color space. LAB第一个通道其实就是灰度图，使用LAB通道进行网络训练时，输入的是lightness channel，即tensor shape为(batch_size, width, height, 1)；输出为剩下的两个channel，即tensor shape为(batch_size, width, height, 2)

The Lab color space describes mathematically all perceivable colors in the three dimensions L for lightness and a and b for the color components green–red and blue–yellow.

关于使用lab color space进行预处理的资料可以参考[about LAB color space 1](https://www.kaggle.com/preslavrachev/wip-photo-colorization-using-keras)中预处理的部分以及[about LAB color space 2](https://fairyonice.github.io/Color-space-defenitions-in-python-RGB-and-LAB.html)

#### 数据集
使用的是cifar数据，训练和测试的时候使用的数据都是彩色的而不是黑白数据!!

### Prediction
进行测试的脚本在两个jupyter noteboook中，以batch的方式进行测试，每个batch的结果存成一张图片，每行是一个sample的结果，以【灰度图，生成图，原始图】的方式写入。

下面给出的是第一个batch使用unet（左）和cgan（右）的结果

![unet prediction](./results/unet/Unet_0.jpg) 
![gan prediction](./results/gan/G_0.jpg) 