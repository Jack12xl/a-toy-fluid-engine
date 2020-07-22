dim = 2
res = [600, 600]
dx = 1.0
inv_dx = 1.0 / dx
half_inv_dx = 0.5 * inv_dx
dt = 0.03
p_jacobi_iters = 30
f_strength = 10000.0
dye_decay = 0.99
debug = False

scheme_setting = dict(
    dim = dim,
    res = res,
    dx = dx,
    inv_dx = inv_dx,
    half_inv_dx = half_inv_dx,

    dt = dt,
    p_jacobi_iters = p_jacobi_iters,
    f_strength = f_strength,
    dye_decay = dye_decay,
    debug = debug,

)
