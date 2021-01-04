from .fluidCFG import FluidCFG
from utils import SetterProperty
from config.class_cfg import SceneEnum, VisualizeEnum, SchemeType, SimulateType
from projection import RedBlackGaussSedialProjectionSolver, JacobiProjectionSolver, ConjugateGradientProjectionSolver
import taichi as ti
from advection import MacCormackSolver
from utils import SetterProperty, plyWriter
import os


class EulerCFG(FluidCFG):
    """
    Hold property especially for Euler-based simulation
    """

    def __init__(self, cfg):
        super(EulerCFG, self).__init__(cfg)

        self.v_grid_type = cfg.v_grid_type

        self.grid = None
        from Grid import GRIDTYPE
        from DataLayout import collocatedGridData, MacGridData, BimMocqGridData
        if self.v_grid_type == GRIDTYPE.CELL_GRID:
            self.grid = collocatedGridData
        elif self.v_grid_type == GRIDTYPE.FACE_GRID:
            self.grid = MacGridData
        elif self.v_grid_type == GRIDTYPE.Bimocq_GRID:
            self.grid = BimMocqGridData

        # Bimocq
        self.blend_coefficient = None
        self.vel_remap_threshold = None
        self.sclr_remap_threshold = None
        self.vel_remap_frequency = None
        self.sclr_remap_frequency = None

        self.run_scheme = cfg.run_scheme
        # Bimocq

        self.fluid_color = cfg.fluid_color

        self.Colliders = cfg.Colliders
        self.Emitters = cfg.Emitters

        self.VisualType = cfg.VisualType
        self.SceneType = cfg.SceneType

        # set these if simulate the gas
        self.GasAlpha = None
        self.GasBeta = None
        self.GasInitAmbientT = None
        self.GasMaxT = None
        self.SimType = cfg.SimType

        self.advection_solver = cfg.advection_solver
        self.projection_solver = cfg.projection_solver
        self.viscosity_coefficient = cfg.dynamic_viscosity_coefficient

        self.dye_decay = cfg.dye_decay
        self.curl_strength = cfg.curl_strength

        self.grid_save_frequency = None
        self.grid_save_dir = None
        self.bool_save_grid = cfg.bool_save_grid if hasattr(cfg, 'bool_save_grid') else False

    @property
    def run_scheme(self):
        return self._run_scheme

    @run_scheme.setter
    def run_scheme(self, s: SchemeType):
        self._run_scheme = s
        if s == SchemeType.Bimocq:
            self.blend_coefficient = self.cfg.blend_coefficient
            self.vel_remap_threshold = self.cfg.vel_remap_threshold
            self.sclr_remap_threshold = self.cfg.sclr_remap_threshold
            self.vel_remap_frequency = self.cfg.vel_remap_frequency
            self.sclr_remap_frequency = self.cfg.sclr_remap_frequency

    @property
    def SimType(self):
        return self._simulation_type

    @SimType.setter
    def SimType(self, T: SimulateType):
        self._simulation_type = T
        if T == SimulateType.Gas:
            self.GasAlpha = self.cfg.GasAlpha
            self.GasBeta = self.cfg.GasBeta
            self.GasInitAmbientT = self.cfg.GasInitAmbientT
            self.GasMaxT = self.cfg.GasMaxT

    @property
    def SceneType(self):
        return self._scene_type

    @SceneType.setter
    def SceneType(self, T: SceneEnum):
        self._scene_type = T
        if T == SceneEnum.MouseDragDye:
            self.force_radius = self.cfg.force_radius
            self.inv_force_radius = 1.0 / self.cfg.force_radius
            self.f_strength = self.cfg.f_strength
            self.inv_dye_denom = self.cfg.inv_dye_denom

    @property
    def advection_solver(self):
        return self._advection_solver

    @advection_solver.setter
    def advection_solver(self, solver):
        self._advection_solver = solver
        self.semi_order = self.cfg.semi_order
        if self._advection_solver == MacCormackSolver:
            self.macCormack_clipping = self.cfg.macCormack_clipping

    @property
    def projection_solver(self):
        return self._projection_solver

    @projection_solver.setter
    def projection_solver(self, solver):
        self._projection_solver = solver
        if self._projection_solver != ConjugateGradientProjectionSolver:

            self.p_jacobi_iters = self.cfg.p_jacobi_iters
            self.poisson_pressure_alpha = ti.static(- self.dx * self.dx)
            # 1 / 4 for 2D, 1 / 6 for 3D
            self.poisson_pressure_beta = ti.static(1.0 / (2 * self.dim))

            # viscosity
            self.viscosity_coefficient = self.cfg.dynamic_viscosity_coefficient

            # TODO currently hard code the viscosity to be same as pressure
            self.poisson_viscosity_alpha = self.poisson_pressure_alpha
            self.poisson_viscosity_beta = self.poisson_pressure_beta

        else:
            # TODO MGPCG
            pass

    @property
    def viscosity_coefficient(self):
        return self._viscosity_coefficient

    @viscosity_coefficient.setter
    def viscosity_coefficient(self, v):
        # GPU GEM
        self._viscosity_coefficient = v
        self.poisson_viscosity_alpha = self.dx * self.dx / (self.dt * self._viscosity_coefficient)
        self.poisson_viscosity_beta = 1.0 / (self.poisson_viscosity_alpha + 4)

    @SetterProperty
    def bool_save(self, save):
        self.__dict__['bool_save'] = save
        print(">>>>>>>>>>")
        if save:
            self.save_what = self.cfg.save_what
            self.save_frame_length = self.cfg.save_frame_length
            print("Here we will save: ")
            for save_thing in self.save_what:
                self.video_managers.append(ti.VideoManager(
                    output_dir=os.path.join(self.cfg.save_path, str(save_thing)),
                    framerate=self.cfg.frame_rate,
                    automatic_build=False
                )
                )
                print(str(save_thing), end=" ")
            print("")
            print("for {} frame with {} Frame Per Second".format(self.save_frame_length, self.cfg.frame_rate))
            print("When done, plz go to {} for results !".format(self.cfg.save_path))

            self.bool_save_ply = self.cfg.bool_save_ply
        else:
            print("Won't save results to disk this time !")
        print(">>>>>>>>>>")

    @SetterProperty
    def bool_save_ply(self, save):
        self.__dict__['bool_save_ply'] = save
        print(">>>>>>")
        if save:
            print("We will save ply every {} !".format(self.ply_frequency))
            self.PLYwriter = plyWriter(self)
            self.ply_frequency = self.cfg.ply_frequency
            print("When done, plz refer to {}".format(self.PLYwriter.series_prefix))
        print(">>>>>>")

    @SetterProperty
    def bool_save_grid(self, save):
        self.__dict__['bool_save_grid'] = save
        print(">>>>>>")
        if save:
            self.grid_save_frequency = self.cfg.grid_save_frequency
            self.grid_save_dir = self.cfg.grid_save_dir
            print("We will save grid info every {} frame!".format(self.grid_save_frequency))
        print(">>>>>>")
