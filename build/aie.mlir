module {
  aie.device(npu2_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<128xi32>> 
    aie.objectfifo @out(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4096xi32>> 
    func.func private @aie_add_wahbm(memref<4096xi32>, i32, memref<128xi32>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<128xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
        %2 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<4096xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<4096xi32>> -> memref<4096xi32>
        %c4096_i32 = arith.constant 4096 : i32
        %c128_i32 = arith.constant 128 : i32
        func.call @aie_add_wahbm(%3, %c4096_i32, %1, %c128_i32) : (memref<4096xi32>, i32, memref<128xi32>, i32) -> ()
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    } {link_with = "add_wahbm.cc.o"}
    aiex.runtime_sequence(%arg0: memref<128xi32>, %arg1: memref<4096xi32>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<128xi32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<4096xi32>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
  }
}

