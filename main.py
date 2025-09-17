from random import sample, randint
from skimage import data, transform, color, io
from numpy import hstack, vstack, zeros
import matplotlib.pyplot as plt


# Набор встроенных изображений из skimage
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


def random_images(images, min_n=4, max_n=9):
    """
    Возвращает случайный список изображений.

    :param images: список доступных изображений
    :param min_n: минимальное количество изображений
    :param max_n: максимальное количество изображений
    :return: список случайных изображений
    """
    if not images:
        raise ValueError("Список изображений пуст!")

    if min_n > max_n:
        raise ValueError("min_n не может быть больше max_n")

    n = randint(min_n, min(max_n, len(images)))
    return sample(images, n)


def make_collage(images, grid_size=(3, 3), img_size=(128, 128)):
    """
    Создаёт коллаж из набора изображений.

    :param images: список изображений
    :param grid_size: размер сетки (строки, столбцы), например (3, 3)
    :param img_size: размер каждого изображения (высота, ширина)
    :return: итоговое изображение-коллаж (numpy-массив)
    """
    rows, cols = grid_size
    resized_images = []

    for img in images:
        # Преобразование ч/б и многоканальных изображений в RGB
        if img.ndim == 2:  # grayscale
            img = color.gray2rgb(img)
        elif img.shape[2] > 3:  # RGBA или больше
            img = img[:, :, :3]

        # Масштабирование
        resized = transform.resize(img, img_size, anti_aliasing=True)
        resized_images.append(resized)

    # Если изображений меньше, чем нужно для сетки — добавляем пустые
    total_cells = rows * cols
    while len(resized_images) < total_cells:
        resized_images.append(zeros((*img_size, 3)))

    # Сборка коллажа
    collage_rows = []
    for r in range(rows):
        row = hstack(resized_images[r * cols:(r + 1) * cols])
        collage_rows.append(row)

    return vstack(collage_rows)


if __name__ == "__main__":
    try:
        # Выбираем случайные изображения
        selected_images = random_images(images, 4, 9)

        # Создаём коллаж (например, 3x3 из картинок 128x128)
        res_collage = make_collage(selected_images, grid_size=(3, 3), img_size=(128, 128))

        # Сохраняем результат
        io.imsave('collage.png', (res_collage * 255).astype('uint8'))

        # Отображаем
        plt.imshow(res_collage)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Ошибка: {e}")
