import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev', default='npu2')
parser.add_argument('--emit-mlir', action='store_true')
parser.add_argument('--xclbin-path')
parser.add_argument('--insts-path')
args = parser.parse_args()
print('''module {
  aie.device(npu2_1col) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    aie.objectfifo @x(%shim, {%core}, 4 : i32) : !aie.objectfifo<memref<16xf32>>
    aie.objectfifo @out(%core, {%shim}, 1 : i32) : !aie.objectfifo<memref<1xf32>>
    func.func private @compute(memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<1xf32>) attributes {link_with = "kernel.o"}
    %worker = aie.core(%core) {
      %a = aie.objectfifo.acquire @x(Consume, 4) : !aie.objectfifosubview<memref<16xf32>>
      %a0 = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<16xf32>> -> memref<16xf32>
      %a1 = aie.objectfifo.subview.access %a[1] : !aie.objectfifosubview<memref<16xf32>> -> memref<16xf32>
      %a2 = aie.objectfifo.subview.access %a[2] : !aie.objectfifosubview<memref<16xf32>> -> memref<16xf32>
      %a3 = aie.objectfifo.subview.access %a[3] : !aie.objectfifosubview<memref<16xf32>> -> memref<16xf32>
      %b = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
      %v = aie.objectfifo.subview.access %b[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
      func.call @compute(%a0, %a1, %a2, %a3, %v) : (memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<16xf32>, memref<1xf32>) -> ()
      aie.objectfifo.release @x(Consume, 4)
      aie.objectfifo.release @out(Produce, 1)
      aie.end
    }
    aie.runtime_sequence(%x: memref<64xf32>, %out: memref<1xf32>) {
      %tx = aiex.dma_configure_task_for @x {
        aie.dma_bd(%x : memref<64xf32> offset = 0 len = 64 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1])
        aie.end
      }
      aiex.dma_start_task(%tx)
      %tout = aiex.dma_configure_task_for @out {
        aie.dma_bd(%out : memref<1xf32> offset = 0 len = 1 sizes = [1, 1, 1, 1] strides = [0, 0, 0, 1])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tout)
      aiex.dma_await_task(%tout)
      aiex.dma_free_task(%tx)
      aiex.dma_free_task(%tout)
    }
  }
}''')
