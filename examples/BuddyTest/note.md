- `math.rsqrt` 报错 `op was not bufferized`

> https://mlir.llvm.org/docs/Bufferization/#using-one-shot-bufferize
> By default, One-Shot Bufferize fails when it encounters an op with tensor semantics (i.e., tensor result or tensor operand) that is not bufferizable (i.e., does not implement BufferizableOpInterface). This can be avoided with allow-unknown-ops. In that case, One-Shot Bufferize inserts to_memref/to_tensor ops around the bufferization boundary.

查看`math.rsqrt` 是否实现 `BufferizableOpInterface`

可行的解决方案：
- `allow-unknown-ops`
- 使用`dialect-filter` 限制 bufferized的范围， eg. `dialect-filter=scf`

1. ../../build/bin/buddy-opt resnet.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" | ../../llvm/build/bin/mlir-opt -one-shot-bufferize="bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map allow-unknown-ops" -convert-linalg-to-parallel-loops -canonicalize -gpu-map-parallel-loops -convert-parallel-loops-to-gpu -gpu-kernel-outlining -canonicalize -cse  -o tmp.mlir

2.  ../../llvm/build/bin/mlir-opt tmp.mlir  -gpu-kernel-outlining -llvm-request-c-wrappers -convert-vector-to-scf -convert-vector-to-llvm > after_gpu_outlining.mlir

3. ../../llvm/build/bin/mlir-opt \
    -pass-pipeline "builtin.module(
    nvvm-attach-target{chip=sm_75 O=3},  
    gpu.module(
        convert-scf-to-cf, 
        convert-gpu-to-nvvm, 
        convert-arith-to-llvm
    ), 
    convert-scf-to-cf, 
    gpu-to-llvm, 
    reconcile-unrealized-casts, 
    gpu-module-to-binary
)" after_gpu_outlining.mlir

1. ../../build/bin/buddy-opt resnet.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named))"
2. ../../build/bin/buddy-opt 1.mlir -pass-pipeline "builtin.module(func.func(tosa-to-linalg))"
3. ../../build/bin/buddy-opt 2.mlir -pass-pipeline "builtin.module(func.func(tosa-to-tensor))"
4. ../../build/bin/buddy-opt 3.mlir -pass-pipeline "builtin.module(func.func(tosa-to-arith))"
5. ../../llvm/build/bin/mlir-opt 4.mlir -one-shot-bufferize="allow-unknown-ops bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map"
5. ../../llvm/build/bin/mlir-opt 5.mlir -expand-realloc
5. ../../llvm/build/bin/mlir-opt 6.mlir -ownership-based-buffer-deallocation
7. ../../llvm/build/bin/mlir-opt 7.mlir -canonicalize
8. ../../llvm/build/bin/mlir-opt 8.mlir -buffer-deallocation-simplification
8. ../../llvm/build/bin/mlir-opt 9.mlir -bufferization-lower-deallocations
6. ../../llvm/build/bin/mlir-opt 10.mlir -convert-linalg-to-parallel-loops
<!-- 7. ../../llvm/build/bin/mlir-opt 6.mlir -canonicalize -->
8. ../../llvm/build/bin/mlir-opt 11.mlir -gpu-map-parallel-loops
9. ../../llvm/build/bin/mlir-opt 12.mlir -convert-parallel-loops-to-gpu
10. ../../llvm/build/bin/mlir-opt 13.mlir -gpu-kernel-outlining
11. ../../llvm/build/bin/mlir-opt 14.mlir -canonicalize
12. ../../llvm/build/bin/mlir-opt 15.mlir -cse
13. ../../llvm/build/bin/mlir-opt 16.mlir -llvm-request-c-wrappers
14. ../../llvm/build/bin/mlir-opt 17.mlir -convert-vector-to-scf
15. ../../llvm/build/bin/mlir-opt 18.mlir -convert-vector-to-llvm
16. ../../llvm/build/bin/mlir-opt 19.mlir -pass-pipeline "builtin.module(nvvm-attach-target{chip=sm_75 O=3})"
17. ../../llvm/build/bin/mlir-opt 20.mlir -pass-pipeline "builtin.module(gpu.module(convert-scf-to-cf))"
18. ../../llvm/build/bin/mlir-opt 21.mlir -pass-pipeline "builtin.module(gpu.module(convert-gpu-to-nvvm))"
19. ../../llvm/build/bin/mlir-opt 22.mlir -pass-pipeline "builtin.module(gpu.module(convert-arith-to-llvm))"
20. ../../llvm/build/bin/mlir-opt 23.mlir -pass-pipeline "builtin.module(convert-scf-to-cf)"


21. ../../llvm/build/bin/mlir-opt 24.mlir -pass-pipeline "builtin.module(finalize-memref-to-llvm)"
21. ../../llvm/build/bin/mlir-opt 25.mlir -pass-pipeline "builtin.module(convert-func-to-llvm)"
21. ../../llvm/build/bin/mlir-opt 26.mlir -pass-pipeline "builtin.module(gpu-to-llvm)"
22. 

22. ../../llvm/build/bin/mlir-opt 26.mlir -pass-pipeline "builtin.module(reconcile-unrealized-casts)"
23. ../../llvm/build/bin/mlir-opt 22.mlir -pass-pipeline "builtin.module(gpu-module-to-binary)"
