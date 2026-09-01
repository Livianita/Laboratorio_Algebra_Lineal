import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from dataset import load_images


def main():
    filenames, images = load_images()
    total_to_show = min(9, len(images))
    output_path = 'dataset_preview.png'

    plt.figure(figsize=(8, 8))

    for i in range(total_to_show):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(filenames[i], fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Vista previa guardada en {output_path}')


if __name__ == '__main__':
    main()
