import matplotlib.pyplot as plot


def read_image():
    def aux():
        while True:
            try:
                yield tuple(map(float, input().split()))
            except EOFError:
                break

    return tuple(aux())


plot.axis('off')
plot.imshow(read_image())
plot.show()
