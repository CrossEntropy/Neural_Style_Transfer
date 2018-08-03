import tensorflow as tf
import scipy.io
import scipy.misc
import numpy as np
from skimage import transform

sess = tf.Session()  # 创建会话


class Config:  # 定义Config类，用来存放路径和超参数
    log_dir = "E:\\好东西\\NST\\graph"
    model_path = "E:\\好东西\\NST\\pretrained-model\\imagenet-vgg-verydeep-19.mat"
    save_path = "E:\\好东西\\NST\\output\\"
    mean = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    noise_ratio = 0.6
    width = 400
    height = 300
    channels = 3
    style_layers = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]
    alpha = 10
    beta = 40
    learning_rate = 2.0
    num_epochs = 200


def load_vgg_model(model_path=Config.model_path):
    vgg = scipy.io.loadmat(model_path)  # 加载模型
    vgg_layers = vgg["layers"]  # 将VGG模型中的layers拿出来，用了来提取权重

    def _weights(layer, expected_layer_name):
        """
        提取VGG模型中的权重和偏置
        :param layer: int, 层数
        :param expected_layer_name: str， 确保你提取层的名字和你希望的相同
        :return: W, np.ndarray is shape of (1, 3, 3, n_C)
                 b, np.ndarray is shape of (n_C, 1)
        """
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    def _conv2d(A_prev, layer, layer_name):
        """
        进行卷积操作
        :param A_prev: tf.tensor is shape of (1, n_H_prev, n_W_prev, n_C_prev) ，上一层的输出
        :param layer: int, 层数
        :param layer_name: str, 层的名字
        :return: tf.tensor is shape of (1, n_H, n_W, n_C)，这一层的输出
        """
        with tf.name_scope(layer_name):
            W, b = _weights(layer, layer_name)
            W = tf.constant(W, name="weights")
            b = tf.constant(np.reshape(b, newshape=(b.size, )), name="biases")
            with tf.name_scope("convolution"):
                Z = tf.add(tf.nn.conv2d(A_prev, W, strides=[1, 1, 1, 1], padding="SAME"), b, name="Z")
            with tf.name_scope("activation"):
                A = tf.nn.relu(Z, name="relu")
            return A

    def _avg_pool(A, num):
        """
        执行平均池化操作
        :param A:  tf.tensor is shape of (1, n_H，n_W. n_C)
        :param num: int, 层的数字
        :return: tf.tensor is shape of (1, n_H. n_W, n_C)
        """
        with tf.name_scope("average_pool"+str(num)):
            return tf.nn.avg_pool(A, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    model = dict()  # 初始化一个字典
    model["input"] = tf.Variable(np.zeros(shape=(1, Config.height, Config.width, Config.channels)), dtype=tf.float32,
                                 name="image")
    model['conv1_1'] = _conv2d(model['input'], 0, 'conv1_1')
    model['conv1_2'] = _conv2d(model['conv1_1'], 2, 'conv1_2')
    model['avgpool1'] = _avg_pool(model['conv1_2'], 1)
    model['conv2_1'] = _conv2d(model['avgpool1'], 5, 'conv2_1')
    model['conv2_2'] = _conv2d(model['conv2_1'], 7, 'conv2_2')
    model['avgpool2'] = _avg_pool(model['conv2_2'], 2)
    model['conv3_1'] = _conv2d(model['avgpool2'], 10, 'conv3_1')
    model['conv3_2'] = _conv2d(model['conv3_1'], 12, 'conv3_2')
    model['conv3_3'] = _conv2d(model['conv3_2'], 14, 'conv3_3')
    model['conv3_4'] = _conv2d(model['conv3_3'], 16, 'conv3_4')
    model['avgpool3'] = _avg_pool(model['conv3_4'], 3)
    model['conv4_1'] = _conv2d(model['avgpool3'], 19, 'conv4_1')
    model['conv4_2'] = _conv2d(model['conv4_1'], 21, 'conv4_2')
    model['conv4_3'] = _conv2d(model['conv4_2'], 23, 'conv4_3')
    model['conv4_4'] = _conv2d(model['conv4_3'], 25, 'conv4_4')
    model['avgpool4'] = _avg_pool(model['conv4_4'], 4)
    model['conv5_1'] = _conv2d(model['avgpool4'], 28, 'conv5_1')
    model['conv5_2'] = _conv2d(model['conv5_1'], 30, 'conv5_2')
    model['conv5_3'] = _conv2d(model['conv5_2'], 32, 'conv5_3')
    model['conv5_4'] = _conv2d(model['conv5_3'], 34, 'conv5_4')
    model['avgpool5'] = _avg_pool(model['conv5_4'], 5)

    return model


def read_image(path):
    """
    读取图像, 并将图像压缩至(300, 400)
    :param path: str，图像存放的路径
    :return: np.ndarray is shape of (300, 400, 3)
    """
    image = scipy.misc.imread(path)
    if image.shape == (300, 400, 3):
        return image
    image = transform.resize(image, (300, 400))
    return image


def reshape_and_normalize(image):
    """
    将300*400图像变为4维数组, 并减去超参数 Config.mean
    :param image: np.ndarray is shape of (300, 400, 3)
    :return: np.ndarray is shape of (1, 300, 400, 3)
    """
    image = np.reshape(image, newshape=(1, ) + image.shape)
    image = image - Config.mean
    return image


def generate_noise_image(content_image, noise_ratio=Config.noise_ratio):
    """
    生成一个带noise的input_image
    :param content_image: np.ndarray is shape of (1, 300, 400, 3)
    :param noise_ratio: int, 0.6
    :return: np.ndarray is hape of (1, 300, 400, 3)
    """

    noise_image = np.random.uniform(-20, 20, size=content_image.shape).astype(dtype=np.float32)
    generated_image = noise_ratio * noise_image + (1-noise_ratio) * content_image
    return generated_image


def save_image(path, image):
    """
    保存图像
    :param path: str,保存图像的路径
    :param image: np.ndarray is shape of (1, 300, 400, 3)
    :return: np.ndarray is shape of (1, 300, 400, 3)
    """
    image = image + Config.mean
    image = np.clip(image[0], 0, 255).astype(dtype=np.uint8)
    scipy.misc.imsave(path, image)
    return image


def content_cost(a):
    """
    计算 content cost
    :param a: tf.tensor is shape of (1, n_H, n_W, n _C)
    :return: tf.tensor is a scalar
    """
    with tf.name_scope("content_cost"):
        m, n_H, n_W, n_C = a.get_shape().as_list()
        a_C = tf.constant(sess.run(a), dtype=tf.float32, name="a_C")
        a_G = a
        cost = tf.reduce_sum((a_G-a_C)**2) / (4 * n_H * n_W * n_C)
        return cost


def gram_matrix(v, ):
    """
    计算 gram_matrix
    :param v: tf.tensor is shape of (n_C, n_H*n_W)
    :return: tf.tensor is shape of (n_C, n_C)
    """
    gm = tf.matmul(v, tf.transpose(v))
    return gm


def single_layer_style_cost(a, layer):
    """
    计算 单层的 style cost
    :param a: tf.tensor is shape of (1, n_H, n_W, n_C)
    :param layer: str, 层的名字
    :return: tf.tensor is a scalar
    """
    with tf.name_scope(layer+"_style_cost"):
        m, n_H, n_W, n_C = a.get_shape().as_list()
        a_S = tf.constant(sess.run(a), dtype=tf.float32, name="a_S")
        a_G = a
        with tf.name_scope("gram_matrix"):
            g_S = gram_matrix(tf.transpose(tf.reshape(a_S, (n_W*n_H, n_C))))
            g_G = gram_matrix(tf.transpose(tf.reshape(a_G, (n_W*n_H, n_C))))
        with tf.name_scope("style_cost"):
            cost = tf.reduce_sum((g_S - g_G)**2) / (4 * (n_H * n_W *n_C)**2)
            return cost


def style_cost(model, style_layers=Config.style_layers):
    """
    计算模型的 style cost
    :param model: dict, 模型
    :param style_layers: list, 存储着名字和系数
    :return: tf.tensor is a scalar
    """
    with tf.name_scope("style_cost"):
        total_cost = 0
        for layer_name,  coeff in style_layers:
            a = model[layer_name]
            cost = single_layer_style_cost(a, layer_name)
            total_cost = total_cost + coeff * cost
        return total_cost


def total_cost(content_cost, style_cost, alpha=Config.alpha, beta=Config.beta):
    """
    计算总的cost
    :param content_cost: tf.tensor is a scalar
    :param style_cost:  tf.tensor is a scala
    :param alpha: int
    :param beta: int
    :return: tf.tensor is a scalar
    """
    with tf.name_scope("total_cost"):
        cost = alpha * content_cost + beta *style_cost
        return cost


def neural_style_transfer(content_path, style_path, save_path=Config.save_path, num_epochs=400, learning_rate=Config.learning_rate,
                          save_graph=False, log_dir=Config.log_dir):
    """
    nst函数
    :param content_path: str, content image path
    :param style_path: str, style image path
    :param save_path: str, save path
    :param num_epochs: int
    :param learning_rate: int
    :param save_graph: bool
    :param log_dir: str
    :return: np.ndarray is shape of (300, 400, 3),最终的生成图像
    """
    assert isinstance(content_path, str)
    assert isinstance(style_path, str)
    assert isinstance(save_path, str)

    # 读取图像
    content_image = read_image(content_path)
    style_image = read_image(style_path)
    # 转换图像，并生成input_image
    content_image = reshape_and_normalize(content_image)
    style_image = reshape_and_normalize(style_image)
    input_image = generate_noise_image(content_image)

    # 构建计算图
    # 加载整个模型，用一个字典来映射模型的各个层
    VGG_19 = load_vgg_model()
    # 首先来计算content cost
    sess.run(VGG_19["input"].assign(content_image))
    J_content = content_cost(VGG_19["conv4_2"])
    # 再来计算style cost
    sess.run(VGG_19["input"].assign(style_image))
    J_style = style_cost(VGG_19)
    # 计算总的 cost
    J_total = total_cost(J_content, J_style)
    # 定义优化节点
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(J_total)

    if save_graph:  # 将计算图保存到event中
        writer = tf.summary.FileWriter(logdir=log_dir, graph=tf.get_default_graph())
        writer.close()

    # 初始化节点
    sess.run(tf.global_variables_initializer())
    # 将 input image 载入
    sess.run(VGG_19["input"].assign(input_image))
    # 运行
    for epoch in range(num_epochs):
        sess.run(train_op)
        image = sess.run(VGG_19["input"])
        if epoch % 20 == 0:
            J_t, J_c, J_s = sess.run([J_total, J_content, J_style])
            print("Epoch is " + str(epoch))
            print("total cost = " + str(J_t))
            print("content cost = " + str(J_c))
            print("style cost = " + str(J_s))
            save_image(save_path + str(epoch) + ".jpg", image)
    image = sess.run(VGG_19["input"])
    save_image(save_path + "generated.jpg", image)
    return image


if __name__ == "__main__":
    content_path = "C:\\python_programme\\Andrew_Ng\\CLASS_4\\week4\\NST\\images\\louvre_small.jpg"
    style_path = "C:\\python_programme\\Andrew_Ng\\CLASS_4\\week4\\NST\\images\\monet.jpg"
    image = neural_style_transfer(content_path, style_path)
