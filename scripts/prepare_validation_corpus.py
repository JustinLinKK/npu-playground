"""Materialize audited scalar/graph baselines for the existing static corpus, without a provider."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


INTEL_MODEL = '''import numpy as np
import openvino as ov
from openvino import opset13 as o


def build_model(manifest):
    nodes = []
    for t in manifest['tensors']:
        if t['direction'] == 'input':
            node = o.parameter(t['shape'], np.dtype(t['dtype']), name=t['name'])
            node.output(0).get_tensor().set_names({t['name']})
            nodes.append(node)
    x = nodes[0]
    p = manifest['oracle']['parameters']
    op = manifest['oracle']['operation']
    f32 = lambda v: o.convert(v, np.float32)
    c = lambda v: o.constant(v, np.float32)
    if op == 'vector_add':
        y = o.add(*nodes)
    elif op == 'reduce_sum':
        y = o.reduce_sum(x, o.constant([p.get('axis', 0)], np.int64), keep_dims=p.get('keepdims', False))
    elif op == 'matmul':
        y = o.convert(o.matmul(f32(x), f32(nodes[1]), False, False), np.float16)
    elif op == 'transpose':
        y = o.transpose(x, o.constant(p['axes'], np.int64))
    elif op == 'softmax':
        y = o.softmax(x, p.get('axis', -1))
    elif op == 'layer_norm':
        axis = o.constant([-1], np.int64)
        shifted = o.subtract(x, o.gather(x, o.constant([0], np.int64), o.constant(-1, np.int64)))
        centered = o.subtract(shifted, o.reduce_mean(shifted, axis, True))
        variance = o.reduce_mean(o.multiply(centered, centered), axis, True)
        y = o.add(o.multiply(o.divide(centered, o.sqrt(o.add(variance, c(p['epsilon'])))), nodes[1]), nodes[2])
    elif op == 'attention':
        scores = o.multiply(o.matmul(f32(x), f32(nodes[1]), False, True), c(p['scale']))
        y = o.convert(o.matmul(o.softmax(scores, -1), f32(nodes[2]), False, False), np.float16)
    elif op == 'moving_average':
        n = manifest['tensors'][0]['shape'][0]
        window = p['window']
        shaped = o.reshape(x, o.constant([1, 1, n], np.int64), False)
        weights = o.constant(np.ones((1, 1, window), dtype=np.float32))
        sums = o.convolution(shaped, weights, [1], [window - 1], [0], [1])
        y = o.divide(o.reshape(sums, o.constant([n], np.int64), False), c(np.minimum(np.arange(n) + 1, window)))
    elif op == 'conv2d':
        y = o.convolution(x, nodes[1], [1, 1], [p['padding']] * 2, [p['padding']] * 2, [1, 1])
    elif op == 'gamma_correction':
        y = o.power(o.clamp(x, 0., 1.), c(p['gamma']))
    elif op == 'sigmoid':
        y = o.sigmoid(x)
    else:
        raise ValueError(op)
    y.output(0).get_tensor().set_names({manifest['tensors'][-1]['name']})
    return ov.Model([y], nodes)
'''


def amd_source(manifest):
    op = manifest['oracle']['operation']
    shapes = [t['shape'] for t in manifest['tensors']]
    params = manifest['oracle']['parameters']
    size = 1
    for d in shapes[0]:
        size *= d
    types = {'float32': 'float', 'float16': 'npu::float16'}
    names = ['x', 'b', 'v'][:len(shapes) - 1] + ['out']
    args = ', '.join(types[t['dtype']] + ' *' + name for t, name in zip(manifest['tensors'], names))
    if op == 'vector_add':
        body = f'for(int i=0;i<{size};++i) out[i]=x[i]+b[i];'
    elif op == 'reduce_sum':
        body = f'float sum=0; for(int i=0;i<{size};++i) sum+=x[i]; out[0]=sum;'
    elif op == 'matmul':
        m,k = shapes[0]; n = shapes[1][1]
        body = f'for(int i=0;i<{m};++i) for(int j=0;j<{n};++j) {{ float sum=0; for(int k=0;k<{k};++k) sum+=(float)x[i*{k}+k]*(float)b[k*{n}+j]; out[i*{n}+j]=sum; }}'
    elif op == 'transpose':
        m,n=shapes[0]
        body=f'for(int i=0;i<{m};++i) for(int j=0;j<{n};++j) out[j*{m}+i]=x[i*{n}+j];'
    elif op == 'softmax':
        n=shapes[0][-1]
        body=f'for(int row=0;row<{size//n};++row) {{ float maximum=x[row*{n}], sum=0; for(int j=1;j<{n};++j) maximum=npu::maximum(maximum,x[row*{n}+j]); for(int j=0;j<{n};++j) sum+=npu::exp(x[row*{n}+j]-maximum); for(int j=0;j<{n};++j) out[row*{n}+j]=npu::exp(x[row*{n}+j]-maximum)/sum; }}'
    elif op == 'layer_norm':
        n=shapes[0][-1]; eps=params['epsilon']
        body=f'for(int row=0;row<{size//n};++row) {{ float anchor=x[row*{n}], mean=0, variance=0; for(int j=0;j<{n};++j) mean+=(x[row*{n}+j]-anchor)/{n}; for(int j=0;j<{n};++j) {{ float z=(x[row*{n}+j]-anchor)-mean; variance+=z*z/{n}; }} float denom=npu::sqrt(variance+{eps}f); for(int j=0;j<{n};++j) out[row*{n}+j]=((x[row*{n}+j]-anchor)-mean)/denom*b[j]+v[j]; }}'
    elif op == 'attention':
        batches,heads,n,d=shapes[0]; scale=params['scale']
        body=f'for(int h=0;h<{batches*heads};++h) for(int i=0;i<{n};++i) {{ float scores[{n}], maximum=-1e30f, denom=0; for(int j=0;j<{n};++j) {{ float sum=0; for(int k=0;k<{d};++k) sum+=(float)x[(h*{n}+i)*{d}+k]*(float)b[(h*{n}+j)*{d}+k]; scores[j]=sum*{scale}f; maximum=npu::maximum(maximum,scores[j]); }} for(int j=0;j<{n};++j) {{ scores[j]=npu::exp(scores[j]-maximum); denom+=scores[j]; }} for(int k=0;k<{d};++k) {{ float sum=0; for(int j=0;j<{n};++j) sum+=(scores[j]/denom)*(float)v[(h*{n}+j)*{d}+k]; out[(h*{n}+i)*{d}+k]=sum; }} }}'
    elif op == 'moving_average':
        window=params['window']
        body=f'for(int i=0;i<{size};++i) {{ int start=i-{window}+1; if(start<0) start=0; float sum=0; for(int j=start;j<=i;++j) sum+=x[j]; out[i]=sum/(i-start+1); }}'
    elif op == 'conv2d':
        batches,channels,height,width=shapes[0]; outs,_,kh,kw=shapes[1]; pad=params['padding']
        body=f'for(int n=0;n<{batches};++n) for(int o=0;o<{outs};++o) for(int r=0;r<{height};++r) for(int s=0;s<{width};++s) {{ float sum=0; for(int c=0;c<{channels};++c) for(int i=0;i<{kh};++i) for(int j=0;j<{kw};++j) {{ int y=r+i-{pad}, z=s+j-{pad}; if(y>=0 && y<{height} && z>=0 && z<{width}) sum+=x[((n*{channels}+c)*{height}+y)*{width}+z]*b[((o*{channels}+c)*{kh}+i)*{kw}+j]; }} out[((n*{outs}+o)*{height}+r)*{width}+s]=sum; }}'
    elif op == 'gamma_correction':
        gamma=params['gamma']
        body=f'for(int i=0;i<{size};++i) out[i]=npu::pow(npu::minimum(1,npu::maximum(0,x[i])),{gamma}f);'
    elif op == 'sigmoid':
        body=f'for(int i=0;i<{size};++i) {{ float e=npu::exp(-npu::absolute(x[i])); out[i]=x[i]>=0 ? 1/(1+e) : e/(1+e); }}'
    else:
        raise ValueError(op)
    if len(shapes) == 4:
        count = 1
        for dimension in shapes[1]:
            count *= dimension
        args = ', '.join(types[manifest['tensors'][i]['dtype']] + ' *' + names[i] for i in (0, 1, 3))
        body = types[manifest['tensors'][2]['dtype']] + f' *v=b+{count}; ' + body
    return '#include <npu_numeric.h>\nextern "C" void compute('+args+') {\n    '+body+'\n}\n'


def prepare(corpus: Path, destination: Path):
    template = Path(__file__).resolve().parents[1] / 'tests/fixtures/backends/amd_single_worker.py.template'
    manifests = sorted(corpus.glob('*/manifest.json'))
    manifests.append(Path('tests/fixtures/backends/intel_npu_4000/sigmoid/manifest.json'))
    for path in manifests:
        manifest = json.loads(path.read_text())
        for target in ('amd_xdna2_npu2', 'intel_npu_4000'):
            directory = destination / target / manifest['name']
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            if target.startswith('amd'):
                (directory / 'design.py').write_text(template.read_text())
                (directory / 'kernel.cc').write_text(amd_source(manifest))
            else:
                (directory / 'model.py').write_text(INTEL_MODEL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, default=Path('examples/classic'))
    parser.add_argument('--output', type=Path, default=Path('runs/validation-corpus'))
    args = parser.parse_args()
    prepare(args.corpus, args.output)
