import numpy as np
import config.base_cfg as base_cfg

def vec2_npf32(m):
    return np.array([m[0], m[1]], dtype=np.float32)

def npNormalize(a, order=2, axis=0):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis)) + base_cfg.error
    # l2[l2 == 0] = 1
    return a / l2

if __name__ == '__main__':
    A = np.array([[0,0],[0,0]])
    print(A)
    print(npNormalize(A))
    # print(A)
    # print(npNormalize(A, 0))
    # print(npNormalize(A, 1))
    # print(npNormalize(A, 2))