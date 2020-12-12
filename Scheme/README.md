#### Euler Solver Scheme

##### [Advection-Reflection](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiAtZeogrfrAhWIvpQKHRdjAuEQtwIwAXoECAkQAQ&url=https%3A%2F%2Fjzehnder.me%2Fpublications%2FadvectionReflection%2F&usg=AOvVaw12RvEOOxqcZ0C7h5urs7f1)

Both running in RK=3, MacCormack + Gauss-Sedial.

| Advection-Projection               | Advection-Reflection                  |
| ---------------------------------- | ------------------------------------- |
| ![](../results/proj-mc-sd-rk3.gif) | ![](../results/reflect-mc-sd-rk3.gif) |

  

#### Performance Comparison

Todo..



##### IVOCK (2014)

Pause the implementation. Find the stream function way too difficult.

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

In the map case, we are showing the first map. Each pixel value(R, G, B) represents mapping to (R, G)  in **UV** space.

Why the backmap and forward map stay the same and always map to itself?

Failing case: Velocity field would explooooode.  ..

| Velocity field                                           | backward map                                | Forward Map                                 | Distortion                                     |
| -------------------------------------------------------- | ------------------------------------------- | ------------------------------------------- | ---------------------------------------------- |
| ![](../results/BiMocq/failed/bimocq_failed_velocity.gif) | ![](../results/BiMocq/failed/failed_BM.gif) | ![](../results/BiMocq/failed/failed_FM.gif) | ![](../results/BiMocq/failed/failed_dstrt.gif) |

Still In progress.

