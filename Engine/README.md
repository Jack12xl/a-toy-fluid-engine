

Here we have different kinds of fluid solver.

- Euler 
- Material Point method 
- lattice boltzmann method (WIP)

#### Euler Solver Scheme

- MacCormack(2005)
- [Advection-Reflection](https://jzehnder.me/publications/advectionReflection/)(2018)
- BiMocq2(2019)

##### Gallery

| Advection-Projection               | Advection-Reflection                  |
| ---------------------------------- | ------------------------------------- |
| ![](../results/proj-mc-sd-rk3.gif) | ![](../results/reflect-mc-sd-rk3.gif) |


#### Performance Comparison

Todo..

##### BiMocq (2019)

Here I adopt

- CFL = 0.5, 
- dt = 0.03, 
- jet_velocity = 0.5

| Density                             | Divergence                         | Velocity                           |
| ----------------------------------- | ---------------------------------- | ---------------------------------- |
| ![](../results/BiMocq/jet/rho.gif)  | ![](../results/BiMocq/jet/div.gif) | ![](../results/BiMocq/jet/vel.gif) |
| **Vorticity**                       | **Backard Map**                    | **Forward** **Map**                |
| ![](../results/BiMocq/jet/curl.gif) | ![](../results/BiMocq/jet/BM.png)  | ![](../results/BiMocq/jet/BM.png)  |

