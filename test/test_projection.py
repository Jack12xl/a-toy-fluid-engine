# reference Games201
import taichi as ti
import random
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

ti.init()

class ProjectionMethod(Enum):
    Jacobian = 0
    GaussSedial = 1

n = 20

A = ti.field(dtype=ti.f32, shape=(n, n))
x = ti.field(dtype=ti.f32, shape=n)
new_x = ti.field(dtype=ti.f32, shape=n)
b = ti.field(dtype=ti.f32, shape=n)

method = ProjectionMethod.GaussSedial

@ti.kernel
def jacobi_iterate():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] * x[j]

        new_x[i] = r / A[i, i]

    for i in range(n):
        x[i] = new_x[i]

@ti.kernel
def gauss_sedial_iterate():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] * new_x[j]

        new_x[i] = r / A[i, i]

    for i in range(n):
        x[i] = new_x[i]

@ti.kernel
def residual() -> ti.f32:
    res = 0.0

    for i in range(n):
        r = b[i] * 1.0
        for j in range(n):
            r -= A[i, j] * x[j]
        res += r * r

    return res

if __name__ == '__main__':
    ## randomly init
    for i in range(n):
        for j in range(n):
            A[i, j] = random.random() - 0.5

        A[i, i] += n * 0.1

        b[i] = random.random() * 100

    residuals = []
    for i in range(8):
        if (method == ProjectionMethod.Jacobian):
            jacobi_iterate()
        elif (method == ProjectionMethod.GaussSedial):
            gauss_sedial_iterate()

        cur_residual = residual()
        residuals.append(cur_residual)
        print(f'iter {i}, residual={cur_residual:0.10f}')

    # assert
    # for i in range(n):
    #     lhs = 0.0
    #     for j in range(n):
    #         lhs += A[i, j] * x[j]
    #     assert abs(lhs - b[i]) < 1e-4

    # plot
    residuals = np.array(residuals)
    # np.save('./jacobi.npy', residuals)
    plt.plot(residuals, color='green')
    # jacobi_resi = np.load('./jacobi.npy')
    # plt.plot(jacobi_resi, color='red')
    plt.show()

