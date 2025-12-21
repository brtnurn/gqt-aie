import numpy as np
from aie.iron import Program, Runtime, Worker, Kernel, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

dev = NPU2()

def my_add_wahbm():
    num_variant = 1024
    worker_per_col = 4
    column_count = 8
    num_workers = worker_per_col * column_count

    worker_input_size = (num_variant + 31 - 1) // 31 
    worker_output_size = num_variant
    col_input_size = worker_input_size  * worker_per_col
    col_output_size = worker_output_size * worker_per_col
    input_size = worker_input_size  * num_workers
    output_size = worker_output_size * num_workers

    worker_in_ty = np.ndarray[(worker_input_size,), np.dtype[np.uint32]]
    worker_out_ty = np.ndarray[(worker_output_size,), np.dtype[np.uint32]]
    col_in_ty = np.ndarray[(col_input_size,), np.dtype[np.uint32]]
    col_out_ty = np.ndarray[(col_output_size,), np.dtype[np.uint32]]
    in_ty = np.ndarray[(input_size,), np.dtype[np.uint32]]
    out_ty = np.ndarray[(output_size,), np.dtype[np.uint32]]

    ofs_col_in  = [ObjectFifo(col_in_ty,  depth=2, name=f"col{c}_in")  for c in range(column_count)]
    ofs_col_out = [ObjectFifo(col_out_ty, depth=2, name=f"col{c}_out") for c in range(column_count)]

    taps_in = TensorTiler2D.simple_tiler((input_size,), (col_input_size,))
    taps_out = TensorTiler2D.simple_tiler((output_size,), (col_output_size,))
    
    add_fn = Kernel(
        "aie_add_wahbm",
        "add_wahbm.cc.o",
        [worker_out_ty, np.int32, worker_in_ty, np.int32],
    )

    def worker_fn(of_in, of_out, adder):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        adder(elem_out, worker_output_size, elem_in, worker_input_size)
        of_in.release(1)
        of_out.release(1)

    workers = []
    for col in range(column_count):
        in_offsets  = [i * worker_input_size  for i in range(worker_per_col)]
        out_offsets = [i * worker_output_size for i in range(worker_per_col)]

        in_fifos = ofs_col_in[col].cons().split(
            in_offsets,
            obj_types=[worker_in_ty] * worker_per_col,
            names=[f"c{col}_w{i}_in" for i in range(worker_per_col)],
        )

        out_fifos = ofs_col_out[col].prod().join(
            out_offsets,
            obj_types=[worker_out_ty] * worker_per_col,
            names=[f"c{col}_w{i}_out" for i in range(worker_per_col)],
        )

        workers.extend(
            [
                Worker(
                    worker_fn,
                    [
                        in_fifos[row].cons(),
                        out_fifos[row].prod(),
                        add_fn,
                    ],
                    placement=Tile(col, row + 2)
                )
                for row in range(worker_per_col)
            ]
        )

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, a_out):
        rt.start(*workers)
        for c in range(column_count):
            rt.fill(ofs_col_in[c].prod(), a_in, taps_in[c], placement=Tile(c, 0))
            rt.drain(ofs_col_out[c].cons(), a_out, taps_out[c], placement=Tile(c, 0), wait=True)

    my_program = Program(dev, rt)
    module = my_program.resolve_program(SequentialPlacer())
    print(module)

if __name__ == "__main__":
    my_add_wahbm()
