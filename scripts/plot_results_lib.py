import numpy as np
import matplotlib.pyplot as plt


def plot_losses(loss, val_loss):
    """
    Show the loss and the validation loss of a model training
    ========= Parameters =======================
    hist: An array with the data of the training

    ========= Return ===========================
    Show the loss and the validation loss
    """
    print("min loss=", min(loss))
    print("min validation loss=", min(val_loss))

    plt.figure(1, figsize=(10, 10))
    plt.plot(loss, ':', color='blue', label="loss")
    plt.plot(val_loss, '--', color='green', label="val_loss")
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title("Losses and val_losses as function of epochs")
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
    pred: An array with the data predict by a model
    y_test: An array with the expected data

    ================= Return ======================
    Show the plot of the chi2 histogram
    """
    chi2 = np.sum((pred - y_test) ** 2, axis=1)
    plt.figure(1, figsize=(10, 10))
    plt.hist(chi2, color="blue", bins=100, alpha=0.5, label='$\chi^2$')
    plt.title("$\chi^2$")
    plt.legend(loc=1)
    plt.show()
    return


def plot_in2out(pred, y_test):
    """
    ================= Parameters ==================
    pred: A 1D array or a list with the data predict by a model
    y_test: A 1D array or a list with the expected data

    ================= Return ======================
    Show the plot of the prediction as a function of the test data
    """
    plt.figure(1, figsize=(10, 10))
    plt.plot(y_test, pred, ' ', marker='.', label="20ep", color='blue')
    plt.plot(np.linspace(min(y_test), max(y_test), int(max(y_test) - min(y_test)) + 1),
             np.linspace(min(y_test), max(y_test), int(max(y_test) - min(y_test)) + 1), lw=2, label="Expected",
             color='green')
    plt.xlabel("Test data")
    plt.ylabel("Predicted data")
    plt.legend(loc=4)
    plt.show()
    return
