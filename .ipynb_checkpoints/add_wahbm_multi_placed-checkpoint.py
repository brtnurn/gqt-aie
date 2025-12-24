import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_
from aie.helpers.taplib import TensorTiler2D


def my_add_wahbm(num_variant, device_name):
    # Select device based on argument
    if device_name == "npu":
        dev = AIEDevice.npu1
    elif device_name == "npu2":
        dev = AIEDevice.npu2
    else:
        raise ValueError(f"Unknown device: {device_name}")

    # Configuration
    worker_per_col = 4  # npu2 architecture has 4 compute tiles per column
    column_count = 8    # for npu2
    num_workers = worker_per_col * column_count  # 32 total workers

    # Calculate sizes (WAH compression: 31 bits per word)
    worker_input_size = (num_variant + 31 - 1) // 31  # ceil(num_variant / 31)
    worker_output_size = num_variant

    col_input_size = worker_input_size * worker_per_col
    col_output_size = worker_output_size * worker_per_col

    input_size = worker_input_size * num_workers
    output_size = worker_output_size * num_workers

    # Define tensor types
    worker_in_ty = np.ndarray[(worker_input_size,), np.dtype[np.uint32]]
    worker_out_ty = np.ndarray[(worker_output_size,), np.dtype[np.uint32]]
    col_in_ty = np.ndarray[(col_input_size,), np.dtype[np.uint32]]
    col_out_ty = np.ndarray[(col_output_size,), np.dtype[np.uint32]]
    in_ty = np.ndarray[(input_size,), np.dtype[np.uint32]]
    out_ty = np.ndarray[(output_size,), np.dtype[np.uint32]]

    # Lists to hold column-level object fifos (populated during device_body)
    ofs_col_in = []
    ofs_col_out = []

    with mlir_mod_ctx() as ctx:
        # TensorAccessPatterns for data movement from main memory to each column
        taps_in = TensorTiler2D.simple_tiler((input_size,), (col_input_size,))
        taps_out = TensorTiler2D.simple_tiler((output_size,), (col_output_size,))

        @device(dev)
        def device_body():
            # External kernel declaration
            # Signature: (output_buffer, output_size, input_buffer, input_size)
            aie_add_wahbm = external_func(
                "aie_add_wahbm",
                inputs=[worker_out_ty, np.int32, worker_in_ty, np.int32]
            )

            # Create tiles for all columns
            shim_tiles = [tile(col, 0) for col in range(column_count)]
            mem_tiles = [tile(col, 1) for col in range(column_count)]
            compute_tiless = [
                [tile(col, row + 2) for row in range(worker_per_col)]
                for col in range(column_count)
            ]

            # Set up each column
            for col in range(column_count):
                shim_tile = shim_tiles[col]
                mem_tile = mem_tiles[col]
                compute_tiles = compute_tiless[col]

                # Create column-level object fifos (shim <-> mem tile)
                of_col_in = object_fifo(
                    f"col{col}_in", shim_tile, mem_tile, 2, col_in_ty
                )
                of_col_out = object_fifo(
                    f"col{col}_out", mem_tile, shim_tile, 2, col_out_ty
                )

                # Offsets for splitting input and joining output
                in_offsets = [i * worker_input_size for i in range(worker_per_col)]
                out_offsets = [i * worker_output_size for i in range(worker_per_col)]

                # Create worker-level object fifos (mem tile <-> compute tiles)
                in_fifos = [
                    object_fifo(
                        f"c{col}_w{w}_in", mem_tile, compute_tiles[w], 2, worker_in_ty
                    )
                    for w in range(worker_per_col)
                ]
                out_fifos = [
                    object_fifo(
                        f"c{col}_w{w}_out", compute_tiles[w], mem_tile, 2, worker_out_ty
                    )
                    for w in range(worker_per_col)
                ]

                # Link column fifos to worker fifos (split input, join output)
                object_fifo_link(of_col_in, in_fifos, [], in_offsets)
                object_fifo_link(out_fifos, of_col_out, out_offsets, [])

                # Store references for runtime sequence
                ofs_col_in.append(of_col_in)
                ofs_col_out.append(of_col_out)

                # Define core logic for each worker in this column
                for w in range(worker_per_col):
                    @core(compute_tiles[w], "add_wahbm.cc.o")
                    def core_body():
                        for _ in range_(sys.maxsize):
                            elem_in = in_fifos[w].acquire(ObjectFifoPort.Consume, 1)
                            elem_out = out_fifos[w].acquire(ObjectFifoPort.Produce, 1)
                            aie_add_wahbm(
                                elem_out, worker_output_size,
                                elem_in, worker_input_size
                            )
                            in_fifos[w].release(ObjectFifoPort.Consume, 1)
                            out_fifos[w].release(ObjectFifoPort.Produce, 1)

            # Runtime sequence - PARALLEL execution across all columns
            @runtime_sequence(in_ty, out_ty)
            def sequence(a_in, a_out):
                in_tasks = []
                out_tasks = []

                # Start ALL DMA tasks first (all columns in parallel)
                for c, (tap_in, tap_out) in enumerate(zip(taps_in, taps_out)):
                    # Configure and start input DMA for column c
                    in_task = shim_dma_single_bd_task(
                        ofs_col_in[c], a_in, tap_in, issue_token=True
                    )
                    dma_start_task(in_task)

                    # Configure and start output DMA for column c
                    out_task = shim_dma_single_bd_task(
                        ofs_col_out[c], a_out, tap_out, issue_token=True
                    )
                    dma_start_task(out_task)

                    in_tasks.append(in_task)
                    out_tasks.append(out_task)

                # Wait for ALL outputs at once (single synchronization point)
                # This allows all 8 columns to execute in parallel!
                dma_await_task(*out_tasks)
                dma_free_task(*in_tasks)

        # Verify and print the generated MLIR module
        res = ctx.module.operation.verify()
        if res == True:
            print(ctx.module)
        else:
            print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate AIE MLIR for WAH bitmap decompression"
    )
    # Positional argument for device (matches Makefile: python3 $< ${devicename})
    parser.add_argument(
        "device",
        help="The device to generate the IR for (npu or npu2)",
        type=str,
        nargs="?",
        default="npu2"
    )
    parser.add_argument(
        "-n", "--num_variant",
        help="Number of variants per worker (output size). Must match add_wahbm.cc CONST_R_SIZE!",
        type=int,
        default=7440  # 240 × 31 - avoids ObjectFifo issue with last WAH word
    )
    args = parser.parse_args()

    my_add_wahbm(args.num_variant, args.device)

