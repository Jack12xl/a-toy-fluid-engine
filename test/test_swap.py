from utils import bufferPair

def swap(a, b):
    a, b = b, a


class foo():
    def __init__(self, _fields):
        self.fields =  _fields

        self.test_bar1 = [233]
        self.test_bar2 = [233, 2333]

    def swap(self):
        self.test_bar1, self.test_bar2 = self.test_bar2, self.test_bar1



if __name__ == "__main__":
    bar1 = foo([[2, 3], [4, 5], [6, 7]])
    bar2 = foo([[11, 22], [33, 44], [55, 66]])
    advect_bar = bufferPair(bar1, bar2)

    advect_foo = []
    for i in range(3):
        advect_foo.append(bufferPair(bar1.fields[i], bar2.fields[i]))

    for i in range(3):
        advect_foo[i].swap()

    advect_bar.swap()
    advect_bar.cur.fields[1][0] = 333
    advect_foo[1].cur[1] = 444
    print(advect_bar.cur.fields)
    print(advect_foo[1].cur)

    # bar1.swap()
    # print(bar1.test_bar1)

    # a = [2, 3, 3]
    # b = [3, 4, 4]
    # swap(a, b)
    # print(a)
    # print(b)
