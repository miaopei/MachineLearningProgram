import matplotlib.pyplot as plt
from cnn import *
from data_utils import get_CIFAR10_data
from solver import Solver

data = get_CIFAR10_data()
model = ThreeLayerConvNet(reg=0.9)
solver = Solver(model, data,                
                lr_decay=0.95,                
                print_every=10, num_epochs=5, batch_size=2, 
                update_rule='sgd_momentum',                
                optim_config={'learning_rate': 5e-4, 'momentum': 0.9})

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
print 'Validation set accuracy: ', (y_val_pred == data['y_val']).mean()
print 'Test set accuracy: ', (y_test_pred == data['y_test']).mean()
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
