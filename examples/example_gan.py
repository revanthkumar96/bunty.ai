from bforbuntyai import GAN, DCGAN, ConditionalGAN, VAE, dataset

# --- Vanilla GAN on Fashion-MNIST ---
gan = GAN(dataset.FashionMNIST(), latent_dim=100)
gan.train(epochs=50)
gan.generate(n=25)
gan.save("fashion_gan.pth")

# --- DCGAN on MNIST ---
dcgan = DCGAN(dataset.MNIST(), latent_dim=100)
dcgan.train(epochs=20)
dcgan.generate(n=25)

# --- Conditional GAN: one image per digit ---
cgan = ConditionalGAN(dataset.MNIST(), num_classes=10)
cgan.train(epochs=50)
cgan.generate_class(labels=list(range(10)))

# --- VAE with 2-D latent space (MNIST) ---
vae = VAE(dataset.MNIST(), latent_dim=2)
vae.train(epochs=20)
vae.visualize()
vae.visualize_latent()

# --- VAE interpolation on Fashion-MNIST ---
vae20 = VAE(dataset.FashionMNIST(), latent_dim=20)
vae20.train(epochs=15)
x_test, _ = dataset.FashionMNIST().as_numpy("test")
vae20.interpolate(x_test[0], x_test[1], steps=10)
