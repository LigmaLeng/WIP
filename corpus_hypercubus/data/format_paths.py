from .import xp
import struct


class Raw:
    root = "corpus_hypercubus/data/raw/"
    bhola = root + "mycareersfuture.json"
    train = root + "train.csv"
    valid = root + "valid.csv"
    test = root + "test.csv"
    train_labels = root + "train_labels.npy"
    valid_labels = root + "valid_labels.npy"

class Pretrained:
    root = "corpus_hypercubus/data/pretrained/"
    binary = root  + "processing/cc.en.300.bin"
    vectors_extended = root + "processing/extended.npy"
    pca_processed = root + "processing/pca.npz"
    pca_processing = root + "processing/svd.npz"
    vectors_unprocessed = root + "init_vec.npz"
    train = root + "unprocessed/train.npy"
    valid = root + "unprocessed/valid.npy"
    test = root + "unprocessed/test.npy"
    relative_ids = root + "processing/relative_ids.npy"

class Custom:
    root = "corpus_hypercubus/data/custom/"
    vectors_once_reduced = root + "vec.npz"
    vectors_twice_reduced = root + "reduced_vec.npz"
    pca_processed = root + "processing/pca.npz"
    pca_processing = root + "processing/svd.npz"
    train_reduced = root + "partial_process/train.npy"
    valid_reduced = root + "partial_process/valid.npy"
    test_reduced = root + "partial_process/test.npy"
    train_processed = root + "full_process/train.npy"
    valid_processed = root + "full_process/valid.npy"
    test_processed = root + "full_process/test.npy"
    relative_ids = root + "processing/relative_ids.npz"

class Mnist:
    root = "corpus_hypercubus/data/mnist/"
    x_train = root + "train-images-idx3-ubyte"
    y_train = root + "train-labels-idx1-ubyte"
    x_test = root + "t10k-images-idx3-ubyte"
    y_test = root + "t10k-labels-idx1-ubyte"

    @staticmethod
    def load():
        with open(Mnist.x_train, 'rb') as imgs:
            magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
            train_images = xp.fromfile(imgs, dtype=xp.uint8).reshape(num, 1, 28, 28)
        with open(Mnist.y_train, 'rb')as labels:
            magic, n = struct.unpack('>II', labels.read(8))
            train_labels = xp.fromfile(labels, dtype=xp.uint8)
        with open(Mnist.x_test, 'rb') as imgs:
            magic, num, nrows, ncols = struct.unpack('>IIII', imgs.read(16))
            test_images = xp.fromfile(imgs, dtype=xp.uint8).reshape(num, 1, 28, 28)
        with open(Mnist.y_test, 'rb')as labels:
            magic, n = struct.unpack('>II', labels.read(8))
            test_labels = xp.fromfile(labels, dtype=xp.uint8)
        return train_images, train_labels, test_images, test_labels