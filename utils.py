import matplotlib.pyplot as plt

def plotGraph(training, test, metrics):
    plt.plot(range(1,
                   len(training) + 1),
             training,
             label="train",
             c='r',
             marker='.')
    plt.plot(range(1, len(test) + 1), test, label="test", c='b', marker='.')
    plt.xlabel("Epochs")
    plt.ylabel(metrics)
    plt.title("Training & Test " + metrics)
    plt.legend()
    plt.show()