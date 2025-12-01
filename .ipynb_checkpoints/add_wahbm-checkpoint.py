import numpy as np

from aie.iron import Program, Runtime, Worker, Kernel, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2Col1

input_size = 128
output_size = input_size * 32

dev = NPU2Col1()

input_ty = np.ndarray[(input_size,), np.dtype[np.int32]]
output_ty = np.ndarray[(output_size,), np.dtype[np.int32]]

add_fn = Kernel("aie_add_wahbm", "add_wahbm.cc.o", [output_ty, np.int32, input_ty, np.int32])

of_in = ObjectFifo(input_ty, name="in")
of_out = ObjectFifo(output_ty, name="out")

def core_fn(of_in, of_out, add_fn):
    words = of_in.acquire(1)
    out = of_out.acquire(1)
    add_fn(out, output_size, words, input_size)
    of_in.release(1)
    of_out.release(1)

worker_fn = Worker(core_fn, [of_in.cons(), of_out.prod(), add_fn])

rt = Runtime()
with rt.sequence(input_ty, output_ty) as (i, o):
    rt.start(worker_fn)
    rt.fill(of_in.prod(), i)
    rt.drain(of_out.cons(), o, wait=True)

my_program = Program(dev, rt)

module = my_program.resolve_program(SequentialPlacer())

print(module)
