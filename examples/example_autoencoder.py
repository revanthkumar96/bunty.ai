from bforbuntyai import AutoEncoder, dataset

# MNIST AutoEncoder
data = dataset.MNIST()
ae = AutoEncoder(data, encoding_dim=64)
ae.train(epochs=50)
ae.visualize()
ae.save("mnist_autoencoder.keras")

# CIFAR-10 AutoEncoder
data_c = dataset.CIFAR10()
ae_c = AutoEncoder(data_c, encoding_dim=512, loss="mse")
ae_c.train(epochs=30)
ae_c.visualize()
