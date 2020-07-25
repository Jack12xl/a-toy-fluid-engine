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

<a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"><img src="https://github.com/Jack12xl/myFluid/blob/master/results/stable_fluid_demo.gif" height="384px"></a> <a href="https://github.com/Jack12xl/myFluid/blob/master/exp_my_fluid.py"> <img src="https://github.com/Jack12xl/myFluid/blob/master/results/stable_fluid_velocity.gif" height="384px"></a>

Left: simulate adding dye in fluid, Right: velocity.

Implement reference:
[NVIDIA DEVELOP ZONE](https://developer.download.nvidia.cn/books/HTML/gpugems/gpugems_ch38.html),
[taichi official example](https://github.com/taichi-dev/taichi/blob/master/examples/stable_fluid.py)


