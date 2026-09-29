import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dev', default='npu2')
parser.add_argument('--emit-mlir', action='store_true')
parser.add_argument('--xclbin-path')
parser.add_argument('--insts-path')
args = parser.parse_args()
manifest_path = Path('/contract/manifest.json')
if not manifest_path.exists():
    manifest_path = Path(__file__).with_name('manifest.json')
manifest = json.loads(manifest_path.read_text())
n = manifest['tensors'][0]['shape'][0]
tile = 16 if n % 16 == 0 else n
print(f'''module {{
  aie.device(npu2_1col) {{
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)
    %second = aie.tile(0, 3)
    aie.objectfifo @x(%shim, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{tile}xf32>>
    aie.objectfifo @y(%shim, {{%core}}, 2 : i32) : !aie.objectfifo<memref<{tile}xf32>>
    aie.objectfifo @out(%second, {{%shim}}, 2 : i32) : !aie.objectfifo<memref<{tile}xf32>>
    aie.objectfifo @middle(%core, {{%second}}, 2 : i32) : !aie.objectfifo<memref<{tile}xf32>>
    func.func private @copy(memref<{tile}xf32>, memref<{tile}xf32>, i32) attributes {{link_with = "kernel.o"}}
    func.func private @add(memref<{tile}xf32>, memref<{tile}xf32>, memref<{tile}xf32>, i32) attributes {{link_with = "kernel.o"}}
    %worker = aie.core(%core) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %count = arith.constant {n // tile} : index
      %size = arith.constant {tile} : i32
      scf.for %i = %c0 to %count step %c1 {{
        %a = aie.objectfifo.acquire @x(Consume, 1) : !aie.objectfifosubview<memref<{tile}xf32>>
        %av = aie.objectfifo.subview.access %a[0] : !aie.objectfifosubview<memref<{tile}xf32>> -> memref<{tile}xf32>
        %b = aie.objectfifo.acquire @y(Consume, 1) : !aie.objectfifosubview<memref<{tile}xf32>>
        %bv = aie.objectfifo.subview.access %b[0] : !aie.objectfifosubview<memref<{tile}xf32>> -> memref<{tile}xf32>
        %c = aie.objectfifo.acquire @middle(Produce, 1) : !aie.objectfifosubview<memref<{tile}xf32>>
        %cv = aie.objectfifo.subview.access %c[0] : !aie.objectfifosubview<memref<{tile}xf32>> -> memref<{tile}xf32>
        func.call @add(%av, %bv, %cv, %size) : (memref<{tile}xf32>, memref<{tile}xf32>, memref<{tile}xf32>, i32) -> ()
        aie.objectfifo.release @x(Consume, 1)
        aie.objectfifo.release @y(Consume, 1)
        aie.objectfifo.release @middle(Produce, 1)
      }}
      aie.end
    }}
    %worker2 = aie.core(%second) {{
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %count2 = arith.constant {n // tile} : index
      %size2 = arith.constant {tile} : i32
      scf.for %j = %zero to %count2 step %one {{
        %m = aie.objectfifo.acquire @middle(Consume, 1) : !aie.objectfifosubview<memref<{tile}xf32>>
        %mv = aie.objectfifo.subview.access %m[0] : !aie.objectfifosubview<memref<{tile}xf32>> -> memref<{tile}xf32>
        %o = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<{tile}xf32>>
        %ov = aie.objectfifo.subview.access %o[0] : !aie.objectfifosubview<memref<{tile}xf32>> -> memref<{tile}xf32>
        func.call @copy(%mv, %ov, %size2) : (memref<{tile}xf32>, memref<{tile}xf32>, i32) -> ()
        aie.objectfifo.release @middle(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%x: memref<{n}xf32>, %y: memref<{n}xf32>, %out: memref<{n}xf32>) {{
      %tx = aiex.dma_configure_task_for @x {{
        aie.dma_bd(%x : memref<{n}xf32> offset = 0 len = {n} sizes = [1, 1, 1, {n}] strides = [0, 0, 0, 1])
        aie.end
      }}
      aiex.dma_start_task(%tx)
      %ty = aiex.dma_configure_task_for @y {{
        aie.dma_bd(%y : memref<{n}xf32> offset = 0 len = {n} sizes = [1, 1, 1, {n}] strides = [0, 0, 0, 1])
        aie.end
      }}
      aiex.dma_start_task(%ty)
      %tout = aiex.dma_configure_task_for @out {{
        aie.dma_bd(%out : memref<{n}xf32> offset = 0 len = {n} sizes = [1, 1, 1, {n}] strides = [0, 0, 0, 1])
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tout)
      aiex.dma_await_task(%tout)
      aiex.dma_free_task(%tx)
      aiex.dma_free_task(%ty)
      aiex.dma_free_task(%tout)
    }}
  }}
}}''')
