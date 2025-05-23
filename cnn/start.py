#
import matplotlib.pyplot as plt
'''同路径下py模块引用'''

try:
    from . import data_utils
    from . import solver
    from . import cnn
except Exception:
    import data_utils
    import solver
    import cnn

import numpy as np
# 获取样本数据
data = data_utils.get_CIFAR10_data()
# model初始化（权重因子以及对应偏置 w1,b1 ,w2,b2 ,w3,b3，数量取决于网络层数）
model = cnn.ThreeLayerConvNet(reg=0.9)
solver = solver.Solver(model, data,
                lr_decay=0.95,
                print_every=10, num_epochs=1, batch_size=2,          #num_epochs=0.1
                update_rule='sgd_momentum',
                optim_config={'learning_rate': 5e-4, 'momentum': 0.9})
# 训练，获取最佳model
solver.train()

plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
plt.show()


best_model = model
y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print ('Validation set accuracy: ',(y_val_pred == data['y_val']).mean())
print ('Test set accuracy: ', (y_test_pred == data['y_test']).mean())
# Validation set accuracy:  about 52.9%
# Test set accuracy:  about 54.7%


# Visualize the weights of the best network
"""
from vis_utils import visualize_grid

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(3, 32, 32, -1).transpose(3, 1, 2, 0)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
show_net_weights(best_model)
plt.show()
"""