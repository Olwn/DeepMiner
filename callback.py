from keras.callbacks import Callback
import numpy as np


def modify_weights(model):
    weights = model.get_weights()
    for layer, w in enumerate(weights[:1]):
        if len(w.shape) < 4: continue
        w_copy = w.copy()
        print w.shape, w_copy.shape
        k_size = w.shape[1]
        for i in range(1, k_size - 1):
            for j in range(1, k_size - 1):
                i_s = [i - 1, i, i + 1]
                j_s = [j - 1, j, j + 1]
                x = w_copy[np.array(i_s), np.array(j_s)]
                w[i, j] = w_copy[i, j] * 0.5 + 0.5 * np.mean(x, axis=(0, 1))
        weights[layer] = w
    model.set_weights(weights)


class PrintWeights(Callback):
    def __init__(self, model):
        super(PrintWeights, self).__init__()
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0: modify_weights(self.model)

    def on_epoch_end(self, epoch, logs=None):
        if epoch <= 50 and epoch % 10 == 0: modify_weights(model=self.model)
        """
        print '\n'
        for w in self.model.get_weights():
            if len(w.shape) == 4:
                print(w.shape, np.sqrt(np.var(w)), np.mean(w))
        """
