# A toy fluid engine
A repo that tries to reimplement Euler based fluid simulation.

The project is based on [Taichi](https://github.com/taichi-dev/taichi), a programming language that embeds both GPU and CPU  parralleled computing.

### Dependency Installation 

```bash
pip install taichi taichi_glsl
```

##### Fast Run

```bash
python exp_my_fluid.py
```

#### Feature:

- Euler-based Scheme
  - Advection-Projection
  - Advection-Reflection
- Advection
  - Semi-Lagrangian
  - MacCormack / BFECC
- Projection
  - Jacobian
  - Gauss-Seidel 
- Two way Coupling
  - In progress...



#### [Stable fluid (Siggraph 1999) ](https://dl.acm.org/doi/pdf/10.1145/311535.311548)

| Density                                                      | Velocity                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <a href="./exp_my_fluid.py"><img src="./results/stable_fluid_demo.gif" height="384px"> | <a href=",/exp_my_fluid.py"> <img src="./results/stable_fluid_velocity.gif" height="384px"> |








#### Coupling with moving solids(Incomplete)

<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"> <img src="./results/naive_collision.gif" height="384px"></a>

Currently, I'm working on coupling with moving objects. I haven't consider the boundary velocity and pressure change caused by moving objects yet.

#### Simple Show Case

AP = advection-projection

AR = advection-reflection

SL == Semi-Lagrangian

MC == MacCormack

JC == Jacobi Iteration

GS == Gauss Sedial

it == iteration



| AP + SL(RK2) + JC(30 it)           | AP + SL(RK3) + JC(30 it)           | AP + MC(RK3) + JC(30 it)          |
| ---------------------------------- | ---------------------------------- | --------------------------------- |
| ![](results/proj-sl-jc-rk2.gif)    | ![](./results/proj-sl-jc-rk3.gif)  | ![](./results/proj-mc-jc-rk3.gif) |
| **AR + MC(RK3) + SD(30 it)**       | **AR + MC(RK3) + SD(30 it)**       | **AP + MC(RK3) + SD(30 it)**      |
| ![](results/reflect-sl-sd-rk3.gif) | ![](results/reflect-mc-sd-rk3.gif) | ![](./results/proj-mc-sd-rk3.gif) |

The above results are showing the density.

#### Mumbled Comparison

- [Advection](./advection/)
- [Projection](./projection/)

- [Euler Solver Scheme](./Scheme)

#### Reference

##### For implementation

- [NVIDIA GPU GEMs](https://developer.download.nvidia.cn/books/HTML/gpugems/gpugems_ch38.html),
- [taichi official example](https://github.com/taichi-dev/taichi/blob/master/examples/stable_fluid.py)
- [Cornell_class_slides](https://www.cs.cornell.edu/~bindel/class/cs5220-s10/slides/lec14.pdf)
-  [offcial tutorial](https://www.bilibili.com/video/BV1ZK411H7Hc?p=4)
-  [Previous Eulerian Fluid Engine With taichi](https://github.com/JYLeeLYJ/Fluid-Engine-Dev-on-Taichi)
-  

##### Paper

- [Efficient and Conservative Fluids with Bidirectional Mapping](https://github.com/ziyinq/Bimocq#efficient-and-conservative-fluids-with-bidirectional-mapping)



