import taichi as ti
from geometry import *

ti.init(ti.cpu, debug=True)

if __name__ == '__main__':
    m_ball = Ball(transform=Transform2(ti.Vector([2.0, 2.0]), localscale=5.0),
                  velocity=Velocity2(
                      velocity_to_world=ti.Vector([2.0, 2.0]),
                      angular_velocity_to_centroid=2.0))
    m_ball.kern_materialize()

    @ti.kernel
    def test_kern():
        print(m_ball.is_inside_world(ti.Vector([2.0, 2.0])))
        print(m_ball.velocity_at_local_point(ti.Vector([1.0, 0.0])))
        print(m_ball.velocity_at_world_point(ti.Vector([7.0, 2.0])))

    test_kern()
