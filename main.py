from random import sample, randint
from skimage import data, transform, color, io
from numpy import hstack, vstack, zeros
import matplotlib.pyplot as plt


size = 128

images = [
    data.astronaut(),
    data.brick(),
    data.camera(),
    data.cell(),
    data.chelsea(),
    data.coffee(),
    data.coins(),
    data.grass(),
    data.gravel(),
    data.horse(),
    data.moon(),
    data.page()
]

def random_images(images):
    randint(4, 9)
    return sample(images, randint(4, 9))

def make_collage(images):
    resized_images = []

    for img in images:
        if img.ndim == 2:
            img = color.gray2rgb(img)
        elif img.shape[2] > 3:
            img = color.gray2rgb(img[:, :, 0])

        resized = transform.resize(img, (size, size))
        resized_images.append(resized)


    while len(resized_images) < 9:
        resized_images.append(zeros((size, size, 3)))


    row1 = hstack(resized_images[0:3])
    row2 = hstack(resized_images[3:6])
    row3 = hstack(resized_images[6:9])

    return vstack([row1, row2, row3])

res_collage = make_collage(random_images(images))
io.imsave('collage.png', (res_collage * 255).astype('uint8'))
plt.imshow(res_collage)
plt.axis('off')
plt.show()
