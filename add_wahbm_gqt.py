"""
AIE add_wahbm kernel generator for GQT integration

This script generates the AIE kernel for a specific number of variants.
The generated kernel will process 32 WAH bitmaps in parallel.

Usage:
    python add_wahbm_gqt.py --num_variants 1000000

The xclbin and instr files need to be generated once per dataset size
(or use a maximum size and pad smaller datasets).
"""

import argparse
import sys
import numpy as np
from aie.iron import Program, Runtime, Worker, Kernel, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern, TensorTiler2D

dev = NPU2()


def generate_add_wahbm(num_variants: int):
    """
    Generate AIE kernel for add_wahbm with specified number of variants.
    
    Args:
        num_variants: Number of variants in the dataset (R_SIZE)
    
    Architecture:
        - 8 columns x 4 workers per column = 32 parallel workers
        - Each worker processes one WAH-compressed bitmap
        - Host sums the 32 outputs
    """
    worker_per_col = 4
    column_count = 8
    num_workers = worker_per_col * column_count  # 32
    
    # Sizes
    # WAH input: at most (num_variants + 30) / 31 words (31 bits per word)
    worker_input_size = (num_variants + 31 - 1) // 31
    worker_output_size = num_variants
    
    col_input_size = worker_input_size * worker_per_col
    col_output_size = worker_output_size * worker_per_col
    input_size = worker_input_size * num_workers
    output_size = worker_output_size * num_workers
    
    # Check memory constraints
    # Each tile has ~64KB. With depth=1, max output per worker ≈ 15000 * 4 = 60KB
    MAX_VARIANTS = 15000
    if num_variants > MAX_VARIANTS:
        print(f"ERROR: num_variants={num_variants} exceeds maximum {MAX_VARIANTS}", file=sys.stderr)
        print(f"  Each AIE tile has ~64KB memory.", file=sys.stderr)
        print(f"  Consider processing in chunks or reducing dataset size.", file=sys.stderr)
        sys.exit(1)
    
    # Print info to stderr so it doesn't mix with MLIR output
    print(f"Generating AIE kernel for {num_variants} variants:", file=sys.stderr)
    print(f"  Workers: {num_workers} (8 columns x 4 per column)", file=sys.stderr)
    print(f"  Input per worker: {worker_input_size} uint32 ({worker_input_size * 4} bytes)", file=sys.stderr)
    print(f"  Output per worker: {worker_output_size} uint32 ({worker_output_size * 4} bytes)", file=sys.stderr)
    print(f"  Total input: {input_size} uint32 ({input_size * 4 / 1024:.1f} KB)", file=sys.stderr)
    print(f"  Total output: {output_size} uint32 ({output_size * 4 / 1024:.1f} KB)", file=sys.stderr)
    
    # Types
    worker_in_ty = np.ndarray[(worker_input_size,), np.dtype[np.uint32]]
    worker_out_ty = np.ndarray[(worker_output_size,), np.dtype[np.uint32]]
    col_in_ty = np.ndarray[(col_input_size,), np.dtype[np.uint32]]
    col_out_ty = np.ndarray[(col_output_size,), np.dtype[np.uint32]]
    in_ty = np.ndarray[(input_size,), np.dtype[np.uint32]]
    out_ty = np.ndarray[(output_size,), np.dtype[np.uint32]]
    
    # FIFOs
    # Note: Each AIE tile has ~64KB memory. With large num_variants, use depth=1
    # to avoid memory overflow. depth=1 means single buffering (slower but fits).
    # Max num_variants with depth=1: ~15000
    # Max num_variants with depth=2: ~7500
    fifo_depth = 1 if num_variants > 7500 else 2
    print(f"  FIFO depth: {fifo_depth} (based on memory constraints)", file=sys.stderr)
    
    ofs_col_in = [ObjectFifo(col_in_ty, depth=fifo_depth, name=f"col{c}_in") 
                  for c in range(column_count)]
    ofs_col_out = [ObjectFifo(col_out_ty, depth=fifo_depth, name=f"col{c}_out") 
                   for c in range(column_count)]
    
    taps_in = TensorTiler2D.simple_tiler((input_size,), (col_input_size,))
    taps_out = TensorTiler2D.simple_tiler((output_size,), (col_output_size,))
    
    # Kernel
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
        in_offsets = [i * worker_input_size for i in range(worker_per_col)]
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
        
        workers.extend([
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
        ])
    
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, a_out):
        rt.start(*workers)
        for c in range(column_count):
            rt.fill(ofs_col_in[c].prod(), a_in, taps_in[c], placement=Tile(c, 0))
            rt.drain(ofs_col_out[c].cons(), a_out, taps_out[c], placement=Tile(c, 0), wait=True)
    
    my_program = Program(dev, rt)
    module = my_program.resolve_program(SequentialPlacer())
    print(module)


def main():
    parser = argparse.ArgumentParser(
        description="Generate AIE add_wahbm kernel for GQT"
    )
    parser.add_argument(
        "--num_variants", "-n",
        type=int,
        default=1024,
        help="Number of variants in the dataset (default: 1024)"
    )
    
    args = parser.parse_args()
    generate_add_wahbm(args.num_variants)


if __name__ == "__main__":
    main()

