import numpy as np

'''
全连接层：矩阵变换，获取对应目标相同的行与列
输入x: 2*32*16*16
输入x_row: 2*8192
超参w：8192*100
输出：矩阵乘法 2*8192 ->8192*100 =>2*100
'''
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    # Reshape x into rows
    N = x.shape[0]
    x_row = x.reshape(N, -1)         # (N,D) -1表示不知道多少列，指定行，就能算出列 = 2 * 32 * 16 * 16/2 = 8192
    out = np.dot(x_row, w) + b       # 点积，(N,M) 2*8192 8192*100 =>2 * 100
    cache = (x, w, b)

    return out, cache
'''
反向传播之affine矩阵变换
根据dout求出dx,dw,db
由 out = w * x =>
dx = dout * w
dw = dout * x
db = dout * 1
因为dx 与 x，dw 与 w，db 与 b 大小（维度）必须相同
dx = dout * wT  矩阵乘法
dw = dxT * dout 矩阵乘法
db = dout 按列求和
'''
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
      dx = dout * w
    - dw: Gradient with respect to w, of shape (D, M)
      dw = dout * x
    - db: Gradient with respect to b, of shape (M,)
      db = dout * 1
    """

    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)                       # (N,D)
    # dx维度必须跟x维度相同
    dx = np.reshape(dx, x.shape)                 # (N,d1,...,d_k)
    # 转换成二维矩阵
    x_row = x.reshape(x.shape[0], -1)            # (N,D)
    dw = np.dot(x_row.T, dout)                   # (D,M)

    db = np.sum(dout, axis=0, keepdims=True)     # (1,M)

    return dx, dw, db

def relu_forward(x):
    """ 激活函数，解决sigmoid梯度消失问题，网络性能比sigmoid更好
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = ReLU(x)
    cache = x

    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout
    dx[x <= 0] = 0

    return dx

def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
         for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
         0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N

    return loss, dx
'''
softmax_loss 求梯度优点: 求梯度运算简单，方便
softmax: softmax用于多分类过程中，它将多个神经元的输出，映射到（0,1）区间内，
可以看成概率来理解，从而来进行多分类。
Si = exp(i)/[exp(j)求和]
softmax_loss：损失函数，求梯度dx必须用到损失函数，通过梯度下降更新超参
Loss = -[Ypred*ln(Sj真实类别位置的概率值)]求和
梯度dx : 对损失函数求一阶偏导
如果 j = i =>dx = Sj - 1
如果 j != i => dx = Sj
'''
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
         0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    '''
     x - np.max(x, axis=1, keepdims=True) 对数据进行预处理，
     防止np.exp(x - np.max(x, axis=1, keepdims=True))得到结果太分散；
     np.max(x, axis=1, keepdims=True)保证所得结果维度不变；
    '''
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    # 计算softmax，准确的说应该是soft，因为还没有选取概率最大值的操作
    probs /= np.sum(probs, axis=1, keepdims=True)
    # 样本图片个数
    N = x.shape[0]
    # 计算图片损失
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    # 复制概率
    dx = probs.copy()
    # 针对 i = j 求梯度
    dx[np.arange(N), y] -= 1
    # 计算每张样本图片梯度
    dx /= N

    return loss, dx

def ReLU(x):
    """ReLU non-linearity."""
    return np.maximum(0, x)
'''
功能：获取图片特征
前向卷积：每次用一个3维的卷积核与图片RGB各个通道分别卷积（卷积核1与R进行点积，卷积核2与G点积，卷积核3与B点积）,
然后将3个结果求和（也就是 w*x ）,再加上 b，就是新结果某一位置输出，这是卷积核在图片某一固定小范围内（卷积核大小）的卷积，
要想获得整个图片的卷积结果，需要在图片上滑动卷积核（先右后下），直至遍历整个图片。
x: 2*3*32*32  每次选取2张图片，图片大小32*32，彩色(3通道)
w: 32*3*7*7   卷积核每个大小是7*7；对应输入x的3通道，所以是3维，有32个卷积核
pad = 3(图片边缘行列补0)，stride = 1(卷积核移动步长)
输出宽*高结果：(32-7+2*3)/1 + 1 = 32
输出大小：2*32*32*32
'''
def conv_forward_naive(x, w, b, conv_param):
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    '''// : 求整型'''
    H_new = 1 + (H + 2 * pad - HH) // stride
    W_new = 1 + (W + 2 * pad - WW) // stride
    s = stride
    out = np.zeros((N, F, H_new, W_new))

    for i in range(N):       # ith image
        for f in range(F):   # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    #print x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s].shape
                    #print w[f].shape
                    #print b.shape
                    #print np.sum((x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]))
                    out[i, f, j, k] = np.sum(x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]) + b[f]

    cache = (x, w, b, conv_param)

    return out, cache

'''
反向传播之卷积：卷积核3*7*7
输入dout:2*32*32*32
输出dx:2*3*32*32
'''
def conv_backward_naive(dout, cache):

    x, w, b, conv_param = cache
    # 边界补0
    pad = conv_param['pad']
    # 步长
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_new = 1 + (H + 2 * pad - HH) // stride
    W_new = 1 + (W + 2 * pad - WW) // stride

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    # 图片个数
    for i in range(N):       # ith image
        # 卷积核滤波个数
        for f in range(F):   # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    # 3*7*7
                    window = x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s]
                    db[f] += dout[i, f, j, k]
                    # 3*7*7
                    dw[f] += window * dout[i, f, j, k]
                    # 3*7*7 => 2*3*38*38
                    dx_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] += w[f] * dout[i, f, j, k]

    # Unpad
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]

    return dx, dw, db
'''
功能：减少特征尺寸大小
前向最大池化：在特征矩阵中选取指定大小窗口，获取窗口内元素最大值作为输出窗口映射值，
先有后下遍历，直至获取整个特征矩阵对应的新映射特征矩阵。
输入x：2*32*32*32
池化参数：窗口：2*2，步长：2
输出窗口宽，高：(32-2)/2 + 1 = 16
输出大小：2*32*16*16
'''
def max_pool_forward_naive(x, pool_param):
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) // s
    W_new = 1 + (W - WW) // s
    out = np.zeros((N, C, H_new, W_new))
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]
                    out[i, j, k, l] = np.max(window)

    cache = (x, pool_param)

    return out, cache

'''
反向传播之池化：增大特征尺寸大小
在缓存中取出前向池化时输入特征，选取某一范围矩阵窗口，
找出最大值所在的位置，根据这个位置将dout值映射到新的矩阵对应位置上，
而新矩阵其他位置都初始化为0.
输入dout:2*32*16*16
输出dx:2*32*32*32
'''
def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) // s
    W_new = 1 + (W - WW) // s
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    # 取前向传播时输入的某一池化窗口
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]
                    # 计算窗口最大值
                    m = np.max(window)
                    # 根据最大值所在位置以及dout对应值=>新矩阵窗口数值
                    # [false,false
                    #  true, false]  * 1 => [0,0
                    #                        1,0]
                    dx[i, j, k*s:HH+k*s, l*s:WW+l*s] = (window == m) * dout[i, j, k, l]

    return dx