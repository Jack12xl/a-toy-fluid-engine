from utils import bufferPair

def swap(a, b):
    a, b = b, a


class foo():
    def __init__(self, _fields):
        self.fields =  _fields

        self.test_bar1 = [233]
        self.test_bar2 = [233, 2333]
        self.b = [233, 233]

    def swap(self):
        self.test_bar1, self.test_bar2 = self.test_bar2, self.test_bar1

    def test_swap(self, M):
        M, self.b = self.b, M
        print(self.b)

if __name__ == "__main__":
    bar1 = foo([[2, 3], [4, 5], [6, 7]])
    # bar2 = foo([[11, 22], [33, 44], [55, 66]])
    # advect_bar = bufferPair(bar1, bar2)
    #
    # advect_foo = []
    # for i in range(3):
    #     advect_foo.append(bufferPair(bar1.fields[i], bar2.fields[i]))
    #
    # for i in range(3):
    #     advect_foo[i].swap()
    #
    # advect_bar.swap()
    # advect_bar.cur.fields[1][0] = 333
    # advect_foo[1].cur[1] = 444
    # print(advect_bar.cur.fields)
    # print(advect_foo[1].cur)

    a = [2,3]
    bar1.test_swap(a)
    bar1.test_swap(a)

