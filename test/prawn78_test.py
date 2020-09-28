import taichi as ti
import math

ti.init(ti.cpu, debug=True)

@ti.data_oriented
class trsfrm:
    def __init__(self, translation=ti.Vector([0.0, 0.0]), orientation=0.0, localscale=1.0):
        '''
        class variable init by ti.Vector
        :param translation:
        :param orientation:
        :param localscale:
        '''
        self.translation = translation
        self.orientation = orientation % (2 * math.pi)
        self.localScale = localscale

    def __repr__(self):
        return '{} ( Trsln : {}, Ornttn: {}, lclScl: {})'.format(
            self.__class__.__name__,
            self.translation,
            self.orientation,
            self.localScale)

@ti.data_oriented
class trsfrm_field:
    def __init__(self):
        '''
        class member init by ti.Vector.field
        '''
        self.translation = ti.Vector.field(2, dtype=ti.f32, shape=[1])

        # self.orientation = orientation % (2 * math.pi)
        # self.localScale = localscale

    def __repr__(self):
        return '{} ( Trsln : {})'.format(
            self.__class__.__name__,
            self.translation,
            )


@ti.kernel
def kern_test_with_input(t_1 : ti.template(), t_2 : ti.template()):
    print("kern with input t1: ", t_1)
    print("kern with input t2: ", t_2[0])

@ti.kernel
def kern_test_without_input():
    print("kern without input t1: ", t1.translation)
    print("kern without input t2: ", t2.translation[0])



@ti.kernel
def kern_add(t:ti.template()):
    t[0] += ti.Vector([2.0, 2.0])
    # print("kern add field", t[0])

t1 = trsfrm()
t2 = trsfrm_field()

while (True):

    t1.translation = t1.translation + ti.Vector([2.0, 2.0])
    print(t1)
    print("py scope print t1: ",t1.translation)

    # increment t2
    kern_add(t2.translation)

    kern_test_without_input()
    # can not capture the incremented t1
    kern_test_with_input(t1.translation, t2.translation)
    print(" ")