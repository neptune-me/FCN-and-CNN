import data_utils
import fc_net
import layer_utils
import layers
import optim
import solver

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# img = Image.open('D:/数据集/train_data/train/1/1.bmp').convert('L')
# img = np.array(img)

def load_character_dataset(root):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(1, 13):
        for j in range(1, 497):
            img_path = root + '/' + str(i) + '/' + str(j) + '.bmp'
            img = Image.open(img_path).convert('L')
            img = np.array(img)
            X_train.append(img)
            y_train.append(i - 1) #类别只能是0-11
        for j in range(497, 621):
            img_path = root + '/' + str(i) + '/' + str(j) + '.bmp'
            img = Image.open(img_path).convert('L')
            img = np.array(img)
            X_test.append(img)
            y_test.append(i - 1)
            
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

X_train, y_train, X_test, y_test = load_character_dataset('D:/数据集/train_data/train')
data = {
    'X_train':X_train,
    'y_train':y_train,
    'X_val':X_test,
    'y_val':y_test
}
weight_scale = 4e-2
learning_rate = 1e-3
model = fc_net.FullyConnectedNet([400, 200, 100, 50], input_dim=1*28*28, num_classes=12,
              weight_scale=weight_scale, dtype=np.float64)
solver = solver.Solver(model, data,
                print_every=500, num_epochs=50, batch_size=25,
                update_rule='sgd',
                optim_config={
                  'learning_rate': learning_rate,
                }
         )
solver.train()

plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('\nRunning on:', device)
#
# if device == 'cuda':
#     device_name = torch.cuda.get_device_name()
#     print('The device name is:', device_name)
#     cap = torch.cuda.get_device_capability(device=None)
#     print('The capability of this device is:', cap, '\n')
























