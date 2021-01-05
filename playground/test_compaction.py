import taichi as ti

ti.init(ti.gpu, make_thread_local=False, make_block_local=False, kernel_profiler=True)

v = ti.field(dtype=ti.f32, shape=[1024, 1024])


@ti.kernel
def test(o: ti.template()) -> ti.f32:
    ret = 0.0
    for I in ti.grouped(o):
        ret += o[I]
    return ret


if __name__ == "__main__":
    v.fill(2.33)
    for _ in range(1024):
        a = test(v)
        # print(a)

    ti.kernel_profiler_print()