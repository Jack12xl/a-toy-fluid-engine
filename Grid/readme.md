# Grid

This folder stores the code for **Grid** module, where the data is stored during simulation.

 

Currently, it features

- Uniform Grid
- Mac(staggered) Grid



**Notice**: Here **v2** basically we change the boundary condition to match the mac grid results.  



##### Here shows a concrete example to show their performance during simulation. 

- 2D-512x512-Density-AR-MCS-RBGSPS-64it-RK3-curl0.0-dt-0.03

- dx = 1.0 / 512

| Uniform Grid                                                 | Uniform Grid **v2**                                          | Mac Grid                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](../results/Grid/2D-512x512-UniformGrid-Density-AR-MCS-RBGSPS-128it-RK3-curl0.0-dt-0.03.gif) | ![](../results/Grid/2D-512x512-UniformGrid-Density-AR-MCS-RBGSPS-128it-RK3-curl0.0-dt-0.03-version2.gif) | ![](../results/Grid/2D-512x512-MacGrid-Density-AR-MCS-RBGSPS-128it-RK3-curl0.0-dt-0.03.gif) |

From my observation, the 2D jet results would look more similar if they are shot from non-boundary position. 

| Uniform Grid                                                 | Mac Grid                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](../results/Grid/2D-512x512-UniformGrid-Density-AR-MCS-RBGSPS-128it-RK3-curl0.0-dt-0.03-non-boundary.gif) | ![](../results/Grid/2D-512x512-MacGrid-Density-AR-MCS-RBGSPS-128it-RK3-curl0.0-dt-0.03-non-boundary.gif) |





- 3D-512x512x256-Density-AP-SLS-JPS-64it-RK3-curl0.0-dt-0.03

| Uniform Grid                                                 | Uniform Grid v2                                              | Mac Grid                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](../results/Grid/3D-512x512x256-UniformGrid-Density-AP-SLS-JPS-64it-RK3-curl0.0-dt-0.03.gif) | ![](../results/Grid/3D-512x512x256-UniformGrid-Density-AP-SLS-JPS-64it-RK3-curl0.0-dt-0.03-version2.gif) | ![](../results/Grid/3D-512x512x256-MacGrid-Density-AP-SLS-JPS-64it-RK3-curl0.0-dt-0.03.gif) |



#### Reference

1.  Harlow, F. H.; J. E. Welch (1965). "Numerical calculation of time-dependent viscous incompressible flow of fluid with a free surface". *[Physics of Fluids](https://en.wikipedia.org/wiki/Physics_of_Fluids)*. **8**: 2182–2189. [doi](https://en.wikipedia.org/wiki/Doi_(identifier)):[10.1063/1.1761178](https://doi.org/10.1063%2F1.1761178).

