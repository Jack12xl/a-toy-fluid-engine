import utils
import config.test_sdf_cfg as m_cfg
from geometry import *
import taichi as ti
import os


ti.init(ti.cpu, debug=True)
pixel = ti.Vector.field(3, dtype=ti.f32, shape=m_cfg.res)

v = 16

@ti.kernel
def paint(translate: ti.template()):
    # print(shape.transform.translation)
    print('kern:', m_implicitBall.transform.translation)
    # m_implicitBall.translateTo(ti.Vector(20.0, 10.0))
    for I in ti.grouped(pixel):
        # pixel[I] = ti.random() * 255
        local_p = m_implicitBall.transform.to_local(I)

        if (m_implicitBall.is_inside_local(local_p)):
            pixel[I] = ti.Vector([0.0, 0.0, 1.0])
        else:
            pixel[I] = ti.Vector([1.0, 0.0, 0.0])
        # pixel[I] = ti.Vector([ti.random(),ti.random(),ti.random()])

test_x = ti.field(dtype=ti.f32, shape=())
test_y = ti.Vector([1.0, 1.0])
@ti.kernel
def test(x:ti.template(), y:ti.template()):
    x[None] += 1.0
    # y = test_y
    y = y + ti.Vector([2.0, 2.0])
    print(test_y)
    # print(x[None])


m_ball = Ball()
m_implicitBall = SurfaceToImplict(m_ball)

m_implicitBall.transform.translation = ti.Vector([256.0, 256.0])
m_implicitBall.transform.localScale = 16.0
gui = ti.GUI('SDF', tuple(m_cfg.res) )
while gui.running:
    test(test_x, test_y)
    if (gui.get_event(ti.GUI.PRESS)):
        e = gui.event
        # print(e.key)
        if e.key == 'w':
            m_implicitBall.transform.translation = m_implicitBall.transform.translation + \
                                                   ti.Vector([0.0, 1.0]) * v
        elif e.key == 's':
            m_implicitBall.transform.translation = m_implicitBall.transform.translation + \
                                                   ti.Vector([0.0, -1.0]) * v
        elif e.key == 'a':
            m_implicitBall.transform.translation = m_implicitBall.transform.translation + \
                                                   ti.Vector([-1.0, 0.0]) * v
        elif e.key == 'd':
            m_implicitBall.transform.translation = m_implicitBall.transform.translation + \
                                                   ti.Vector([1.0, 0.0]) * v
        elif e.key == 'r':
            m_implicitBall.transform.localScale *= 2.0
        elif e.key == 'f':
            m_implicitBall.transform.localScale *= 0.5

    # print(m_implicitBall.transform.translation)
    # print(m_ball.transform.translation)
    paint(m_implicitBall.transform.translation)
    img = pixel.to_numpy()
    gui.set_image(img)
    gui.show()


