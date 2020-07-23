import taichi as ti
import numpy as np
from enum import Enum
import time

ti.init(arch=ti.cpu)
wi = 1.0 / 6
real = ti.f32


class SolverType(Enum):
    jacobi = 1
    Gauss_Seidel = 2
    multigrid = 3


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


@ti.data_oriented
class SmokeSolver:
    def __init__(self, x, y, z):
        self.res = [x, y, z]
        self.resx, self.resy, self.resz = self.res
        self.dx = 1.0
        self.dt = 0.04
        self.inv_dx = 1.0 / self.dx
        self.half_inv_dx = 0.5 * self.inv_dx
        self.p_alpha = -self.dx * self.dx
        self.use_bfecc = True
        self.solver_type = SolverType.jacobi
        self.max_iter = 60

        self._velocities = ti.Vector(3, dt=ti.f32, shape=self.res)
        self._new_velocities = ti.Vector(3, dt=ti.f32, shape=self.res)
        self.velocity_divs = ti.var(dt=ti.f32, shape=self.res)
        self._pressures = ti.var(dt=ti.f32, shape=self.res)
        self._new_pressures = ti.var(dt=ti.f32, shape=self.res)
        self._dens_buffer = ti.var(dt=ti.f32, shape=self.res)
        self._new_dens_buffer = ti.var(dt=ti.f32, shape=self.res)

        self.velocities_pair = TexPair(self._velocities, self._new_velocities)
        self.pressures_pair = TexPair(self._pressures, self._new_pressures)
        self.dens_pair = TexPair(self._dens_buffer, self._new_dens_buffer)

        self._img = ti.Vector(3, dt=ti.f32, shape=(self.resx + self.resx, self.resy))
        self._pos = ti.Vector(3, dt=ti.f32, shape=self.res)

        # <editor-fold desc="MGPCG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!">
        self.use_multigrid = True
        self.N = self.resx
        self.n_mg_levels = 4
        self.pre_and_post_smoothing = 2
        self.bottom_smoothing = 50
        self.dim = 3

        self.N_ext = self.N // 2  # number of ext cells set so that that total grid size is still power of 2
        self.N_tot = 2 * self.N

        # setup sparse simulation data arrays
        self.r = [ti.var(dt=real) for _ in range(self.n_mg_levels)]  # residual
        self.z = [ti.var(dt=real)
                  for _ in range(self.n_mg_levels)]  # M^-1 self.r
        self.x = ti.var(dt=real)  # solution
        self.p = ti.var(dt=real)  # conjugate gradient
        self.Ap = ti.var(dt=real)  # matrix-vector product
        self.alpha = ti.var(dt=real)  # step size
        self.beta = ti.var(dt=real)  # step size
        self.sum = ti.var(dt=real)  # storage for reductions

        indices = ti.ijk if self.dim == 3 else ti.ij
        self.grid = ti.root.pointer(indices, [self.N_tot // 4]).dense(
            indices, 4).place(self.x, self.p, self.Ap)

        for l in range(self.n_mg_levels):
            self.grid = ti.root.pointer(indices,
                                        [self.N_tot // (4 * 2 ** l)]).dense(
                indices,
                4).place(self.r[l], self.z[l])

        ti.root.place(self.alpha, self.beta, self.sum)
        # </editor-fold>

    @ti.func
    def sample(self, qf, u, v, w):
        i, j, k = int(u), int(v), int(w)
        i = max(0, min(self.resx - 1, i))
        j = max(0, min(self.resy - 1, j))
        k = max(0, min(self.resz - 1, k))
        return qf[i, j, k]

    @ti.func
    def lerp(self, vl, vr, frac):
        # frac: [0.0, 1.0]
        return vl + frac * (vr - vl)

    @ti.func
    def trilerp(self, vf, u, v, w):
        s, t, n = u - 0.5, v - 0.5, w - 0.5
        iu, iv, iw = int(s), int(t), int(n)
        fu, fv, fw = s - iu, t - iv, n - iw
        a = self.sample(vf, iu + 0.5, iv + 0.5, iw + 0.5)
        b = self.sample(vf, iu + 1.5, iv + 0.5, iw + 0.5)
        c = self.sample(vf, iu + 0.5, iv + 1.5, iw + 0.5)
        d = self.sample(vf, iu + 1.5, iv + 1.5, iw + 0.5)
        e = self.sample(vf, iu + 0.5, iv + 0.5, iw + 1.5)
        f = self.sample(vf, iu + 1.5, iv + 0.5, iw + 1.5)
        g = self.sample(vf, iu + 0.5, iv + 1.5, iw + 1.5)
        h = self.sample(vf, iu + 1.5, iv + 1.5, iw + 1.5)

        bilerp1 = self.lerp(self.lerp(a, b, fu), self.lerp(c, d, fu), fv)
        bilerp2 = self.lerp(self.lerp(e, f, fu), self.lerp(g, h, fu), fv)
        return self.lerp(bilerp1, bilerp2, fw)

    @ti.func
    def sample_min(self, qf: ti.template(), coord):
        grid = coord * self.inv_dx - ti.Vector([0.5, 0.5, 0.5])
        I = ti.cast(ti.floor(grid), ti.i32)
        min_val = qf[I]
        for i, j, k in ti.ndrange(2, 2, 2):
            min_val = min(min_val, qf[I + ti.Vector([i, j, k])])
        return min_val

    @ti.func
    def sample_max(self, qf: ti.template(), coord):
        grid = coord * self.inv_dx - ti.Vector([0.5, 0.5, 0.5])
        I = ti.cast(ti.floor(grid), ti.i32)
        max_val = qf[I]
        for i, j, k in ti.ndrange(2, 2, 2):
            max_val = max(max_val, qf[I + ti.Vector([i, j, k])])
        return max_val

    @ti.func
    def back_trace_rk2(self, vf: ti.template(), pos, delta_t):
        mid = pos - 0.5 * delta_t * self.trilerp(vf, pos[0], pos[1], pos[2])
        coord = pos - delta_t * self.trilerp(vf, mid[0], mid[1], mid[2])
        return coord

    @ti.kernel
    def advect_semi_l(self, vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
        for i, j, k in vf:
            pos = ti.Vector([i, j, k]) + 0.5 * self.dx
            coord = self.back_trace_rk2(vf, pos, self.dt)
            new_qf[i, j, k] = self.trilerp(qf, coord[0], coord[1], coord[2])

    @ti.kernel
    def advect_bfecc(self, vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
        for I in ti.grouped(vf):
            pos = ti.Vector([I[0], I[1], I[2]]) + 0.5 * self.dx
            coord = self.back_trace_rk2(vf, pos, self.dt)
            x1 = self.trilerp(qf, coord[0], coord[1], coord[2])
            coord2 = self.back_trace_rk2(vf, coord, -self.dt)
            x2 = self.trilerp(qf, coord2[0], coord2[1], coord[2])
            new_qf[I] = x1 + 0.5 * (x2 - qf[I])

            # clipping ????????
            min_val = self.sample_min(qf, coord)
            max_val = self.sample_max(qf, coord)
            if new_qf[I] < min_val or new_qf[I] > max_val:
                new_qf[I] = x1

    @ti.kernel
    def divergence(self, vf: ti.template()):
        for i, j, k in vf:
            vl = self.sample(vf, i - 1, j, k)[0]
            vr = self.sample(vf, i + 1, j, k)[0]
            vb = self.sample(vf, i, j - 1, k)[1]
            vt = self.sample(vf, i, j + 1, k)[1]
            vh = self.sample(vf, i, j, k - 1)[2]
            vq = self.sample(vf, i, j, k + 1)[2]
            vc = self.sample(vf, i, j, k)
            if i == 0:
                vl = -vc[0]
            if i == self.resx - 1:
                vr = -vc[0]
            if j == 0:
                vb = -vc[1]
            if j == self.resy - 1:
                vt = -vc[1]
            if k == 0:
                vh = -vc[2]
            if k == self.resz - 1:
                vq = -vc[2]
            self.velocity_divs[i, j, k] = (vr - vl + vt - vb + vq - vh) * self.half_inv_dx

    @ti.kernel
    def pressure_jacobi(self, pf: ti.template(), new_pf: ti.template()):
        for i, j, k in pf:
            pl = self.sample(pf, i - 1, j, k)
            pr = self.sample(pf, i + 1, j, k)
            pb = self.sample(pf, i, j - 1, k)
            pt = self.sample(pf, i, j + 1, k)
            ph = self.sample(pf, i, j, k - 1)
            pq = self.sample(pf, i, j, k + 1)
            div = self.velocity_divs[i, j, k]
            new_pf[i, j, k] = (pl + pr + pb + pt + ph + pq + self.p_alpha * div) * wi

    @ti.kernel
    def Gauss_Seidel(self, pf: ti.template(), new_pf: ti.template()):
        for i, j, k in pf:
            if (i + j + k) % 2 == 0:
                pl = self.sample(pf, i - 1, j, k)
                pr = self.sample(pf, i + 1, j, k)
                pb = self.sample(pf, i, j - 1, k)
                pt = self.sample(pf, i, j + 1, k)
                ph = self.sample(pf, i, j, k - 1)
                pq = self.sample(pf, i, j, k + 1)
                div = self.velocity_divs[i, j, k]
                new_pf[i, j, k] = (pl + pr + pb + pt + ph + pq + self.p_alpha * div) * wi
        for i, j, k in pf:
            if (i + j + k) % 2 == 1:
                pl = self.sample(new_pf, i - 1, j, k)
                pr = self.sample(new_pf, i + 1, j, k)
                pb = self.sample(new_pf, i, j - 1, k)
                pt = self.sample(new_pf, i, j + 1, k)
                ph = self.sample(new_pf, i, j, k - 1)
                pq = self.sample(new_pf, i, j, k + 1)
                div = self.velocity_divs[i, j, k]
                new_pf[i, j, k] = (pl + pr + pb + pt + ph + pq + self.p_alpha * div) * wi

    @ti.kernel
    def subtract_gradient(self, vf: ti.template(), pf: ti.template()):
        for i, j, k in vf:
            pl = self.sample(pf, i - 1, j, k)
            pr = self.sample(pf, i + 1, j, k)
            pb = self.sample(pf, i, j - 1, k)
            pt = self.sample(pf, i, j + 1, k)
            ph = self.sample(pf, i, j, k - 1)
            pq = self.sample(pf, i, j, k + 1)
            v = self.sample(vf, i, j, k)
            v = v - self.half_inv_dx * ti.Vector([pr - pl, pt - pb, pq - ph])
            vf[i, j, k] = v

    # <editor-fold desc="MGPCG">

    @ti.kernel
    def init(self):
        for i, j, k in self.velocity_divs:
            self.r[0][i, j, k] = - 1.0 * self.velocity_divs[i, j, k]
            self.z[0][i, j, k] = 0.0
            self.Ap[i, j, k] = 0.0
            self.p[i, j, k] = 0.0
            self.x[i, j, k] = 0.0

    @ti.func
    def neighbor_sum(self, x, I):
        ret = 0.0
        for i in ti.static(range(3)):
            offset = ti.Vector.unit(3, i)
            ret += x[I + offset] + x[I - offset]
        return ret

    @ti.kernel
    def compute_Ap(self):
        for I in ti.grouped(self.Ap):
            self.Ap[I] = (2 * 3) * self.p[I] - self.neighbor_sum(
                self.p, I)

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0
        for I in ti.grouped(p):
            self.sum[None] += p[I] * q[I]

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[0][I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.z[0][I] + self.beta[None] * self.p[I]

    @ti.kernel
    def restrict(self, l: ti.template()):
        for I in ti.grouped(self.r[l]):
            res = self.r[l][I] - (2.0 * self.dim * self.z[l][I] -
                                  self.neighbor_sum(self.z[l], I))
            self.r[l + 1][I // 2] += res * 0.5

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for I in ti.grouped(self.z[l]):
            self.z[l][I] = self.z[l + 1][I // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        for I in ti.grouped(self.r[l]):
            if (I.sum()) & 1 == phase:
                self.z[l][I] = (self.r[l][I] + self.neighbor_sum(
                    self.z[l], I)) / (2.0 * self.dim)

    def apply_preconditioner(self):
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l, 1)
                self.smooth(l, 0)

    @ti.kernel
    def jacobi_precondition(self):
        for i, j, k in self.z[0]:
            self.z[0][i, j, k] = self.r[0][i, j, k] * wi

    def mgpcg_run(self):
        self.init()
        self.reduce(self.r[0], self.r[0])
        initial_rTr = self.sum[None]

        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p
        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0])

        self.update_p()

        self.reduce(self.z[0], self.r[0])
        old_zTr = self.sum[None]

        # CG
        for i in range(self.max_iter):
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / pAp

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]
            if rTr < initial_rTr * 1.0e-12:
                print(i)
                break

            # self.z = M^-1 self.r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0])

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / old_zTr

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            # print(f'iter {i}, residual={rTr}')
    # </editor-fold>

    @ti.kernel
    def source(self):
        a1 = self.resx // 2 - 8
        b1 = self.resx // 2 + 8
        c1 = self.resz // 2 - 8
        d1 = self.resz // 2 + 8

        for i, j, k in ti.ndrange((a1, b1), (10, 20), (c1, d1)):
            self._dens_buffer[i, j, k] = 0.5
        for i, j, k in ti.ndrange((a1, b1), (10, 20), (c1, d1)):
            self._velocities[i, j, k] = ti.Vector([0, 50, 0])

    @ti.kernel
    def to_image(self):
        for i, j in ti.ndrange((0, self.resx), (0, self.resy)):
            self._img[i, j] = self._dens_buffer[i, j, self.resz // 2] * ti.Vector([1, 0.8, 0.8])
        for k, l in ti.ndrange((0, self.resz), (0, self.resy)):
            self._img[k + self.resx, l] = self._dens_buffer[self.resx // 2, l, k] * ti.Vector([0.8, 0.9, 1.0])

    def set_solver(self, solver):
        if solver in (1, SolverType.jacobi, "jacobi"):
            self.solver_type = SolverType.jacobi
        elif solver in (2, SolverType.Gauss_Seidel, "Gauss_Seidel"):
            self.solver_type = SolverType.Gauss_Seidel
        elif solver in (3, SolverType.multigrid, "multigrid"):
            self.solver_type = SolverType.multigrid
        else:
            self.solver_type = SolverType.jacobi
            print("No solver type found. Use jacobi instead.")

    def set_bfecc(self, foo):
        self.use_bfecc = True if foo else False

    def set_max_iter(self, val):
        self.max_iter = val

    def step(self):
        if self.use_bfecc:
            self.advect_semi_l(self.velocities_pair.cur, self.velocities_pair.cur, self.velocities_pair.nxt)
            self.advect_bfecc(self.velocities_pair.cur, self.dens_pair.cur, self.dens_pair.nxt)
        else:
            self.advect_semi_l(self.velocities_pair.cur, self.velocities_pair.cur, self.velocities_pair.nxt)
            self.advect_semi_l(self.velocities_pair.cur, self.dens_pair.cur, self.dens_pair.nxt)
        self.velocities_pair.swap()
        self.dens_pair.swap()
        self.divergence(self.velocities_pair.cur)

        if self.solver_type == SolverType.jacobi:
            for _ in range(self.max_iter):
                self.pressure_jacobi(self.pressures_pair.cur, self.pressures_pair.nxt)
                self.pressures_pair.swap()
            self.subtract_gradient(self.velocities_pair.cur, self.pressures_pair.cur)
        elif self.solver_type == SolverType.Gauss_Seidel:
            for _ in range(self.max_iter):
                self.Gauss_Seidel(self.pressures_pair.cur, self.pressures_pair.nxt)
                self.pressures_pair.swap()
            self.subtract_gradient(self.velocities_pair.cur, self.pressures_pair.cur)
        else:
            self.mgpcg_run()
            self.pressures_pair.cur.copy_from(self.x)
            self.subtract_gradient(self.velocities_pair.cur, self.pressures_pair.cur)

    def reset(self):
        self._dens_buffer.fill(0.0)
        self._velocities.fill(0.0)
        self._pressures.fill(0.0)

    @ti.kernel
    def place_pos(self):
        for i, j, k in self._pos:
            self._pos[i, j, k] = 0.01 * ti.Vector([i, j, k])

    def save_ply(self, frame):
        series_prefix = "smoke.ply"
        num_vertices = self.resx * self.resy * self.resz
        self.place_pos()
        np_pos = np.reshape(self._pos.to_numpy(), (num_vertices, 3))
        np_dens = np.reshape(self._dens_buffer.to_numpy(), (num_vertices, 1))
        writer = ti.PLYWriter(num_vertices=num_vertices)
        writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
        writer.add_vertex_alpha(np_dens)
        writer.export_frame_ascii(frame, series_prefix)

    def run(self):
        gui = ti.GUI("smoke_solver", (self.resx + self.resz, self.resy))
        self.reset()
        self.source()
        count = 0
        timer = 0
        while count < 300:
            for e in gui.get_events(ti.GUI.PRESS):
                if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                    exit()
            self.step()
            self.source()
            self.to_image()
            gui.set_image(self._img)
            delta_t = time.time() - timer
            fps = 1/delta_t
            timer = time.time()
            gui.text(content=f'fps: {fps:.1f}', pos=(0, 0.98), color=0xffaa77)
            filename = f'mgpcg_{self.max_iter:02d}iter_{count:05d}.png'
            gui.show()
            self.save_ply(count)
            count += 1


def main():
    smoke_solver = SmokeSolver(160, 256, 160)
    smoke_solver.set_bfecc(True)
    smoke_solver.set_solver(2)
    smoke_solver.set_max_iter(30)
    smoke_solver.run()


if __name__ == '__main__':
    main()