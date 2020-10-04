# myFluid
A repo that trys to reimplement Euler based fluid simluation

The project is based on [Taichi](https://github.com/taichi-dev/taichi), a programming language that embed both gpu and cpu parrellel computing.

### Taichi Installation 

```bash
python3 -m pip install taichi
```

#### [Stable fluid (siggraph 1999) ](https://dl.acm.org/doi/pdf/10.1145/311535.311548)
```bash
python exp_my_fluid.py
```

<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/stable_fluid_demo.gif" height="384px"></a> 
<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"> <img src="https://github.com/Jack12xl/myFluid/blob/master/results/stable_fluid_velocity.gif" height="384px"></a>

Left: simulate adding dye in fluid, Right: velocity.

Implement reference:
[NVIDIA DEVELOP ZONE](https://developer.download.nvidia.cn/books/HTML/gpugems/gpugems_ch38.html),
[taichi official example](https://github.com/taichi-dev/taichi/blob/master/examples/stable_fluid.py)



#### Naive coupling with moving solids(not complete)

<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"> <img src="https://github.com/Jack12xl/a-toy-fluid-engine/blob/dev/results/collider-rflct-lg-gs-rk2.gif" height="384px"></a>

Currently I am working on coupling with moving objects. I haven't consider the boundary velocity and pressure change caused by moving objects yet.

#### Advection Scheme

change this [line](https://github.com/Jack12xl/myFluid/blob/88dcaf7a39cd976756811f1087767bfc4bd59502/exp_my_fluid.py#L2) to 

```python
import config.stable_fluid_fixed as m_cfg
```

Here we try diffierent advection schemes:

implement reference: [offcial tutorial](https://www.bilibili.com/video/BV1ZK411H7Hc?p=4)

framework reference: [Efficient and Conservative Fluids with Bidirectional Mapping](https://github.com/ziyinq/Bimocq#efficient-and-conservative-fluids-with-bidirectional-mapping)

##### Semi-Lagragian 
<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/sl-rk1.gif" height="384px"></a> 
<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/sl-rk2.gif" height="384px"></a> 

Left: rk = 1, Right: rk = 2.

##### [MacCormack](https://link.springer.com/article/10.1007/s10915-007-9166-4)
<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/mc-rk1.gif" height="384px"></a> 
<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/mc-rk2.gif" height="384px"></a> 

Left: rk = 1, Right: rk = 2.

#### Solver Scheme

Above is all about advection-projection.

##### [Advection-Reflection](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiAtZeogrfrAhWIvpQKHRdjAuEQtwIwAXoECAkQAQ&url=https%3A%2F%2Fjzehnder.me%2Fpublications%2FadvectionReflection%2F&usg=AOvVaw12RvEOOxqcZ0C7h5urs7f1)

Both running in RK=2.



<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/rflct-sl-rk2.gif" height="384px"></a> 
<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/rflct-mc-rk2.gif" height="384px"></a> 

Left: Semi-lagrangion, Right: MacCormack



#### Projection:

Above results is all about jacobi iteration projection solver run in 30 iterations.

##### [Red-Black Gauss Seidel projection](https://www.cs.cornell.edu/~bindel/class/cs5220-s10/slides/lec14.pdf)

Ref: [Cornell_class_slides](https://www.cs.cornell.edu/~bindel/class/cs5220-s10/slides/lec14.pdf)

Both run in RK=2, with advection-projection scheme and 30 iterations.

<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/rflct-sl-gs-rk2.gif" height="384px"></a> 
<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/rflct-mc-gs-rk2.gif" height="384px"></a> 

Left: Semi-lagrangion, Right: MacCormack



#### Current problem:
1. Have no idea whether it is implemented in the right way since the basic semi-lagrangian with RK 1 advection would bend too early 0.0

<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/sl-rk1.gif" height="384px"></a> 



