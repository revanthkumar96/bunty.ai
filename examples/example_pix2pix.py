from bforbuntyai import Pix2Pix, dataset

# Edges2Shoes — downloads ~1.5 GB on first run, cached afterwards
data = dataset.Edges2Shoes(image_size=256, batch_size=16)

p2p = Pix2Pix(data, lambda_l1=100)
p2p.train(epochs=5)
p2p.visualize()       # 5 side-by-side comparisons: edge | generated | real
p2p.save("pix2pix_shoes.pth")
