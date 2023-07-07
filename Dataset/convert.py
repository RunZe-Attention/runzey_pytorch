def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):

        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()



# if __name__ == '__main__':
#     #生成train.csv
#     convert(r"自己的路径\train-images.idx3-ubyte", r"自己的路径\train-labels.idx1-ubyte",
#         r"自己的路径\mnist.train.csv", 60000)
#
#     #生成test.csv
#     convert(r"自己的路径\t10k-images.idx3-ubyte", r"自己的路径\t10k-labels.idx1-ubyte",
#         r"自己的路径\mnist_test.csv", 10000)

if __name__ == '__main__':
        convert(r"./data/FashionMNIST/raw/train-images-idx3-ubyte",
            r"./data/FashionMNIST/raw/train-labels-idx1-ubyte",
            r"./data/FashionMNIST/raw/fashion_mnist.train.csv", 60000)

    #convert(r"C:\Users\Lenovo\fashion mnist\t10k-images-idx3-ubyte/t10k-images.idx3-ubyte",
    #        r"C:\Users\Lenovo\fashion mnist/t10k-labels.idx1-ubyte",
    #        r"C:\Users\Lenovo\fashion mnist/fashion mnist_test.csv", 10000)
print("Convert Finished!")

