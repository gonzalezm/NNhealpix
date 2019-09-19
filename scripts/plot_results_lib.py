import numpy as np
import matplotlib.pyplot as plt

"""
This library makes useful plots to study a training. 
"""

def plot_losses(hist):
    """
    Show the loss and the validation loss of a model training
    ========= Parameters =======================
    hist: dict
        History saved during training
    """
    val_loss = hist['val_loss']
    loss = hist['loss']

    print("min loss=", np.min(loss))
    print("min validation loss=", np.min(val_loss))

    plt.figure(1, figsize=(10, 10))
    plt.plot(loss, ':', color='blue', label="loss")
    plt.plot(val_loss, '--', color='green', label="val_loss")
    plt.xlabel('epoch')
    plt.title("Loss and val_loss as function of epochs")
    plt.legend(loc=1)
    plt.yscale('log')
    plt.show()
    return


def plot_error(pred, y_test):
    """
    Calculate and plot the histogram of the error

     ================= Parameters ==================
    pred: An array with the data predict by a model
    y_test: An array with the expected data
            should have the same shape as pred

    ================= Return ======================
    Show the plot of the error histogram
    """
    err = (pred[:, :] - y_test[:, :])  # /y_test[:,:]*100.
    mean_err = np.mean(abs(err))
    print('Mean error: ', mean_err)
    err = np.ravel(err)
    plt.figure(1, figsize=(10, 10))
    plt.hist(err, bins=100, alpha=0.5, label='err', color='blue')
    plt.title("Histogram of the error between the predicted and expected data")
    plt.xlabel("Difference between the predicted and expected data")
    plt.ylabel("Amount of points")
    plt.legend(loc=1)
    plt.show()
    return


def plot_chi2(pred, y_test):
    """
    Calculate and plot the histogram of the chi2.

    ================= Parameters ==================
    pred: array
        Prediction given by model.predict
    y_test: An array with the expected data
    """
    # Make a 1D array
    pred = np.squeeze(pred, axis=1)

    chi2 = np.sum((pred - y_test) ** 2)

    plt.figure(1, figsize=(10, 10))
    plt.hist(chi2, color='b', bins=2, alpha=0.5)
    plt.xlabel('$\chi^2$')
    plt.legend(loc='best')
    plt.show()
    return


def plot_in2out(pred, y_test):
    """
    Show the plot of the prediction as a function of the test data
    ================= Parameters ==================
    pred: array
     Prediction given by model.predict
    y_test: A 1D array or a list with the expected data
    """
    # Make a 1D array
    pred = np.squeeze(pred, axis=1)

    # To plot y = x as a reference
    x = np.linspace(0., np.max(y_test), 2)

    plt.figure(1, figsize=(10, 10))
    plt.plot(y_test, pred, ' ', marker='.', color='blue')
    plt.plot(x, x, lw=2, color='green', label="y = x")
    plt.xlabel("Real")
    plt.ylabel("Prediction")
    plt.legend(loc=4)
    plt.show()
    return
