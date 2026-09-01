from dataset import build_matrix, load_images


def main():
    filenames, images = load_images()
    X = build_matrix(images)

    print('Total Number of Faces: {}'.format(len(filenames)))
    print(images.shape)
    print(X.shape)


if __name__ == '__main__':
    main()
