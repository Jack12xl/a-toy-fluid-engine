from Grid import GRIDTYPE
import taichi as ti
import taichi_glsl as ts
import numpy as np
from config import VisualizeEnum, SceneEnum, SchemeType, SimulateType
from boundary import StdGridBoundaryConditionSolver
from config import PixelType, EulerCFG
from abc import ABCMeta, abstractmethod
from renderer import renderer2D, renderer25D
from utils import getFieldMeanCpu


@ti.data_oriented
class EulerScheme(metaclass=ABCMeta):
    def __init__(self, cfg:EulerCFG, ):
        self.cfg = cfg
        self.dim = cfg.dim

        self.curFrame = 0

        self.grid = cfg.grid(cfg)

        self.boundarySolver = StdGridBoundaryConditionSolver(cfg, self.grid)

        self.advection_solver = self.cfg.advection_solver(cfg, self.grid,
                                                          self.boundarySolver.collider_sdf_field,
                                                          self.boundarySolver.marker_field)
        self.projection_solver = self.cfg.projection_solver(cfg, self.grid,
                                                            self.boundarySolver.marker_field)

        self.emitters = cfg.Emitters

        if self.dim == 2:
            self.renderer = renderer2D(cfg, self.grid)
        elif self.dim == 3:
            self.renderer = renderer25D(cfg, self.grid, self.cfg.res[2] // 2)

        if cfg.v_grid_type == GRIDTYPE.FACE_GRID:
            self.ApplyBuoyancyForce = self.ApplyBuoyancyForceMac
        elif cfg.v_grid_type == GRIDTYPE.CELL_GRID:
            self.ApplyBuoyancyForce = self.ApplyBuoyancyForceUniform

    def get_CFL(self):
        pass

    def advect_velocity(self, dt):
        for v_pair in self.grid.advect_v_pairs:
            self.advection_solver.advect(self.grid.v_pair.cur, v_pair.cur, v_pair.nxt,
                                         dt)

    def advect(self, dt):
        # advect speed should be handled separately
        self.advect_velocity(dt)

        self.advection_solver.advect(self.grid.v_pair.cur, self.grid.density_pair.cur, self.grid.density_pair.nxt,
                                     dt)

        if self.cfg.SimType == SimulateType.Gas:
            self.advection_solver.advect(self.grid.v_pair.cur,
                                         self.grid.t_pair.cur,
                                         self.grid.t_pair.nxt,
                                         dt)
            self.grid.t_pair.swap()
        self.grid.swap_v()
        self.grid.density_pair.swap()

    def externalForce(self, ext_input, dt):
        # TODO
        if self.cfg.SceneType == SceneEnum.MouseDragDye:
            # add impulse from mouse
            self.apply_impulse(self.grid.v_pair.cur, self.grid.density_pair.cur, ext_input, dt)
        elif self.cfg.SceneType == SceneEnum.Jet:
            for emitter in self.emitters:
                emitter.stepEmitForce(
                    self.grid.v_pair.cur,
                    self.grid.density_bffr,
                    dt
                )
        if self.cfg.SimType == SimulateType.Gas:
            # calculate buoyancy
            self.grid.t_ambient[None] = getFieldMeanCpu(self.grid.t_pair.cur.field)
            # self.ApplyBuoyancyForce(dt)

    @ti.kernel
    def ApplyBuoyancyForceMac(self, dt: ti.f32):
        v_y = ti.static(self.grid.v_pair.cur.fields[1])
        t_f = ti.static(self.grid.t_pair.cur)
        d_f = ti.static(self.grid.density_pair.cur)
        D = ti.Vector.unit(self.dim, 1)
        for I in ti.static(v_y):
            f_buoy = - self.cfg.GasAlpha * d_f[I][1] \
                     + self.cfg.GasBeta * (
                             (
                                     t_f.sample(I) + t_f.sample(I - D)
                              )[0] * 0.5 - self.grid.t_ambient
                     )
            v_y[I][0] += f_buoy * dt

    @ti.kernel
    def ApplyBuoyancyForceUniform(self, dt: ti.f32):
        d_f = ti.static(self.grid.density_pair.cur)
        t_f = ti.static(self.grid.t_pair.cur)
        v_f = ti.static(self.grid.v_pair.cur)
        for I in ti.static(v_f):
            f_buoy = - self.cfg.GasAlpha * d_f[I][1] \
                     + self.cfg.GasBeta * (t_f[I][0] - self.grid.t_ambient)
            # print(f_buoy)
            v_f[I].y += f_buoy * dt

    def project(self):
        self.grid.calDivergence(self.grid.v_pair.cur, self.grid.v_divs)

        self.grid.calVorticity(self.grid.v_pair.cur)
        if self.cfg.curl_strength:
            self.enhance_vorticity()

        self.projection_solver.runPressure()
        # self.projection_solver.runViscosity()

    @ti.kernel
    def enhance_vorticity(self):
        # ref: taichi official stable fluid
        # ref2: https://softologyblog.wordpress.com/2019/03/13/vorticity-confinement-for-eulerian-fluid-simulations/
        vf = ti.static(self.grid.v_pair.cur)
        vc = ti.static(self.grid.v_curl)
        for I in ti.grouped(vc.field):
            cc = vc.sample(I)
            force = ts.vecND(self.dim, 0.0)
            for d in ti.static(range(self.dim)):
                D = ti.Vector.unit(self.dim, d)
                c0 = vc.sample(I + D)
                c1 = vc.sample(I - D)
                # TODO refine 3D cases
                force[self.dim - 1 - d] = abs(c0) - abs(c1)
                if not d & 1:
                    # even
                    force[self.dim - 1 - d] *= -1
            # in 2d this is equivalent to :
            # cl = vc.sample(I + ts.D.zy)
            # cr = vc.sample(I + ts.D.xy)
            # cb = vc.sample(I + ts.D.yz)
            # ct = vc.sample(I + ts.D.yx)
            # cc = vc.sample(I)
            # force = ti.Vector([abs(ct) - abs(cb),
            #                    abs(cl) - abs(cr)]).normalized(1e-3)
            force = force.normalized(1e-3)
            force *= self.cfg.curl_strength * cc
            vf[I] = ts.clamp(vf[I] + force * self.cfg.dt, -1e3, 1e3)

    @ti.kernel
    def apply_impulse(self, vf: ti.template(), dyef: ti.template(),
                      imp_data: ti.ext_arr(),
                      dt: ti.template()):

        for I in ti.grouped(vf):
            mdir = ts.vec(imp_data[0], imp_data[1])
            o = ts.vec(imp_data[2], imp_data[3])
            # move to cell center
            # dx, dy = I + 0.5 - o
            # d2 = dx * dx + dy * dy
            d2 = (I - o).norm_sqr()
            # ref: https://developer.download.nvidia.cn/books/HTML/gpugems/gpugems_ch38.html
            # apply the force
            factor = ti.exp(-d2 * self.cfg.inv_force_radius)
            momentum = mdir * self.cfg.f_strength * dt * factor

            vf[I] += momentum
            # add dye
            dc = dyef[I]
            # TODO what the hell is this?
            if mdir.norm() > 0.5:
                dc += ti.exp(-d2 * self.cfg.inv_dye_denom) * ti.Vector(
                    [imp_data[4], imp_data[5], imp_data[6]])
            dc *= self.cfg.dye_decay
            dyef[I] = dc

    def step(self, ext_input: np.array):

        self.boundarySolver.step_update_sdfs(self.boundarySolver.colliders)
        self.boundarySolver.kern_update_marker()

        # if we don't want do advection projection on emitter area
        # for emitter in self.emitters:
        #     self.boundarySolver.updateEmitterMark(emitter)

        for colld in self.boundarySolver.colliders:
            colld.surfaceshape.update_transform(self.cfg.dt)

        # do advection projection here
        self.schemeStep(ext_input)

        self.boundarySolver.ApplyBoundaryCondition()

        self.dye_fade()
        # refill the fluid
        for emitter in self.emitters:
            emitter.stepEmitHardCode(self.grid.v_pair.cur, self.grid.density_pair.cur, self.grid.t_pair.cur)

        self.renderer.renderStep(self.boundarySolver)
        self.curFrame += 1

    @abstractmethod
    def schemeStep(self, ext_input: np.array):
        pass

    def materialize(self):
        """
        Serve as the initialization process
        Init the field variable
        :return:
        """
        self.materialize_collider()
        self.materialize_emitter()
        self.grid.materialize()

        self.curFrame = 0

    def materialize_emitter(self):
        for emitter in self.emitters:
            emitter.kern_materialize()
            # init the density and velocity for advection
            emitter.stepEmitHardCode(self.grid.v_pair.cur, self.grid.density_pair.cur, self.grid.t_pair.cur)

    def materialize_collider(self):
        for collid in self.boundarySolver.colliders:
            collid.kern_materialize()

    def reset(self):
        self.grid.reset()
        self.renderer.clr_bffr.fill(ti.Vector([0, 0, 0]))
        self.boundarySolver.reset()

        self.curFrame = 0

    @ti.kernel
    def dye_fade(self):
        d = ti.static(self.grid.density_bffr)
        for I in ti.grouped(d.field):
            d[I] *= self.cfg.dye_decay
