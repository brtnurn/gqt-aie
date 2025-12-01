module attributes {llvm.target_triple = "aie2p"} {
  llvm.mlir.global external @in_cons_buff_1() {addr_space = 0 : i32} : !llvm.array<128 x i32>
  llvm.mlir.global external @in_cons_buff_0() {addr_space = 0 : i32} : !llvm.array<128 x i32>
  llvm.mlir.global external @out_buff_1() {addr_space = 0 : i32} : !llvm.array<4096 x i32>
  llvm.mlir.global external @out_buff_0() {addr_space = 0 : i32} : !llvm.array<4096 x i32>
  llvm.func @debug_i32(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.event(i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.put.ms(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.get.ss() -> !llvm.struct<(i32, i32)> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.mcd.write.vec(vector<16xi32>, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.scd.read.vec(i32) -> vector<16xi32> attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.acquire(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @llvm.aie2p.release(i32, i32) attributes {sym_visibility = "private"}
  llvm.func @aie_add_wahbm(!llvm.ptr, i32, !llvm.ptr, i32) attributes {sym_visibility = "private"}
  llvm.func @core_0_2() {
    %0 = llvm.mlir.addressof @in_cons_buff_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @out_buff_1 : !llvm.ptr
    %2 = llvm.mlir.addressof @in_cons_buff_0 : !llvm.ptr
    %3 = llvm.mlir.addressof @out_buff_0 : !llvm.ptr
    %4 = llvm.mlir.constant(2 : index) : i64
    %5 = llvm.mlir.constant(9223372036854775806 : index) : i64
    %6 = llvm.mlir.constant(51 : i32) : i32
    %7 = llvm.mlir.constant(48 : i32) : i32
    %8 = llvm.mlir.constant(50 : i32) : i32
    %9 = llvm.mlir.constant(49 : i32) : i32
    %10 = llvm.mlir.constant(1 : i32) : i32
    %11 = llvm.mlir.constant(-1 : i32) : i32
    %12 = llvm.mlir.constant(128 : i32) : i32
    %13 = llvm.mlir.constant(4096 : i32) : i32
    %14 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%14 : i64)
  ^bb1(%15: i64):  // 2 preds: ^bb0, ^bb2
    %16 = llvm.icmp "slt" %15, %5 : i64
    llvm.cond_br %16, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @llvm.aie2p.acquire(%9, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %11) : (i32, i32) -> ()
    %17 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4096 x i32>
    %18 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<128 x i32>
    llvm.call @aie_add_wahbm(%17, %13, %18, %12) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2p.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%6, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%9, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %11) : (i32, i32) -> ()
    %19 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4096 x i32>
    %20 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<128 x i32>
    llvm.call @aie_add_wahbm(%19, %13, %20, %12) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2p.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%6, %10) : (i32, i32) -> ()
    %21 = llvm.add %15, %4 : i64
    llvm.br ^bb1(%21 : i64)
  ^bb3:  // pred: ^bb1
    llvm.call @llvm.aie2p.acquire(%9, %11) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.acquire(%8, %11) : (i32, i32) -> ()
    %22 = llvm.getelementptr %3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4096 x i32>
    %23 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<128 x i32>
    llvm.call @aie_add_wahbm(%22, %13, %23, %12) : (!llvm.ptr, i32, !llvm.ptr, i32) -> ()
    llvm.call @llvm.aie2p.release(%7, %10) : (i32, i32) -> ()
    llvm.call @llvm.aie2p.release(%6, %10) : (i32, i32) -> ()
    llvm.return
  }
}

