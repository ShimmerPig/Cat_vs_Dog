# 用人话跟你讲TensorFlow猫狗二分类</br>
代码详解：</br>
</br>
1.处理数据</br>
（1）通过get_files()函数读取图片，然后根据图片名，添加猫狗label，再分别将猫狗的image以及label放到四个数组cats，label_cats，dogs，label_dogs中。
将猫狗的image一起放入image_list数组中，并将猫狗的label一起放到label_list数组中。然后将所有的image以及label都放到一个数组中，打乱顺序后再将无序的image_list和label_list返回。</br>
（2）将第一步处理好的image_list和label_list两个数组转化为 tensorflow 能够识别的格式，然后将图片裁剪和补充进行标准化处理，分批次返回。
这里的处理过程如下：读取图片的全部信息-->把图片解码（channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度）-->把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度-->对数据进行标准化,标准化，就是减去它的均值，除以他的方差。
</br></br>
2.设计卷积神经网络</br>
网络结构如下：
# conv1   卷积层 1
# pooling1_lrn  池化层 1
# conv2  卷积层 2
# pooling2_lrn 池化层 2
# local3 全连接层 1
# local4 全连接层 2
# softmax 全连接层 3</br>
2.1卷积层的构建</br>
（1）定义weight与biases变量</br>
在卷积层1中，定义weight与biases变量如下：
        weights = tf.get_variable('weights',  
                                  shape=[3, 3, 3, 16],  
                                  dtype=tf.float32,  
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))  
        biases = tf.get_variable('biases',  
                                 shape=[16],  
                                 dtype=tf.float32,  
                                 initializer=tf.constant_initializer(0.1))  
（2）定义卷积：</br>
tf.nn.conv2d函数是tensoflow里面的二维的卷积函数，第一个参数是图片的所有参数，第二个参数是此卷积层的权重，然后定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME。
conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')  

（3）添加偏差项</br>
这里我们使用bias_add函数将偏差项biases添加到conv上面。
pre_activation = tf.nn.bias_add(conv, biases)  
（5）加上激活函数</br>
我们对conv 进行非线性处理，也就是激活函数来处理，这里我们用的是tf.nn.relu（修正线性单元）来处理，要注意的是，因为采用了SAME的padding方式，输出图片的大小没有变化，只是厚度变厚了。
conv1 = tf.nn.relu(pre_activation, name=scope.name) </br>
2.2池化层的构建</br>
（1）定义池化pooling</br>
为了得到更多的图片信息，padding时我们选的是一次一步，也就是strides[1]=strides[2]=1，这样得到的图片尺寸没有变化，而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。pooling 有两种，一种是最大值池化，一种是平均值池化，这里采用的是最大值池化tf.max_pool()。池化的核函数大小为3X3，因此ksize=[1,3,3,1]，步长为2，因此strides=[1,2,2,1]:
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')  
（2）局部响应归一化</br>
局部响应归一化原理是仿造生物学上活跃的神经元对相邻神经元的抑制现象（侧抑制）。
LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。

norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')  
2.3建立全连接层    </br>
进入全连接层时, 我们通过tf.reshape()将pool2的输出值从一个三维的变为一维的数据，将上一个输出结果展平。
reshape = tf.reshape(pool2, shape=[batch_size, -1])  
然后将展平后的reshape与本层的weights相乘
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)  </br>
2.4分类器</br>
这里我们用softmax分类器（多分类，输出是各个类的概率）,对我们的输出进行分类。
    # softmax  
    with tf.variable_scope('softmax_linear') as scope:  
        weights = tf.get_variable('softmax_linear',  
                                  shape=[128, n_classes],  
                                  dtype=tf.float32,  
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))  
        biases = tf.get_variable('biases',  
                                 shape=[n_classes],  
                                 dtype=tf.float32,  
                                 initializer=tf.constant_initializer(0.1))  
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')  
  
    return softmax_linear </br>
2.5计算误差</br>
这里我们使用了sparse_softmax_cross_entropy_with_logits，并且利用交叉熵损失函数来定义我们的cost function。
def losses(logits, labels):  
    with tf.variable_scope('loss') as scope:  
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
                        (logits=logits, labels=labels, name='xentropy_per_example')  
        loss = tf.reduce_mean(cross_entropy, name='loss')  
        tf.summary.scalar(scope.name + '/loss', loss)  
    return loss </br>
2.6选优化方法</br>
我们用tf.train.AdamOptimizer()作为我们的优化器进行优化。
optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)  
</br></br>
3.训练数据，并将训练的模型存储</br>
这里我们定义了两个输出神经元，［1，0］ 或者 ［0，1］表示猫和狗的概率。
N_CLASSES = 2 
并且重新定义了图片的大小--因为图片如果过大的话会导致训练比较慢。
IMG_W = 208  
IMG_H = 208
定义每批数据的大小为32
BATCH_SIZE = 32
训练的步数为15000
MAX_STEP = 15000
学习效率为0.0001
learning_rate = 0.0001
训练的大致过程如下：
获取图片和标签集-->生成批次-->进入模型-->获取loss（损失）-->训练-->合并summary-->保存summary</br></br>
4.对模型进行测试</br>
由于每次只读取一张图片，所以将batch设置为1，输出神经元同上。这里需要对图片进行处理，具体步骤如下：转换图片格式-->图片标准化-->三维转四维-->最终的结果用softmax激活。通过从指定的路径加载模型进行测试。</br></br>
