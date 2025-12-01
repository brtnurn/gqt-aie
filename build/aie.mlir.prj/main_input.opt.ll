; ModuleID = '/notebooks/bertan/gqt/build/aie.mlir.prj/main_input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@in_cons_buff_1 = external global [128 x i32]
@in_cons_buff_0 = external global [128 x i32]
@out_buff_1 = external global [4096 x i32]
@out_buff_0 = external global [4096 x i32]

; Function Attrs: nounwind
declare void @llvm.aie2p.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2p.release(i32, i32) #0

declare void @aie_add_wahbm(ptr, i32, ptr, i32) local_unnamed_addr

define void @core_0_2() local_unnamed_addr {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %3, %1 ]
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @aie_add_wahbm(ptr nonnull @out_buff_0, i32 4096, ptr nonnull @in_cons_buff_0, i32 128)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @aie_add_wahbm(ptr nonnull @out_buff_1, i32 4096, ptr nonnull @in_cons_buff_1, i32 128)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  %3 = add nuw nsw i64 %2, 2
  %.not = icmp eq i64 %3, 9223372036854775806
  br i1 %.not, label %4, label %1

4:                                                ; preds = %1
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @aie_add_wahbm(ptr nonnull @out_buff_0, i32 4096, ptr nonnull @in_cons_buff_0, i32 128)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  ret void
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
