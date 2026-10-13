"""Prepare thirty CUDA adaptations of a fixed KernelBench revision; no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np

from npu_agent.kernelbench import reference
from npu_agent.models import KernelManifest


COMMIT = '423217d9fda91e0c2d67e4a43bf62f96f6d104f1'
BASE = f'https://raw.githubusercontent.com/ScalingIntelligence/KernelBench/{COMMIT}'
ACTIVATIONS = {19: 'relu', 20: 'leaky_relu', 21: 'sigmoid', 22: 'tanh', 24: 'log_softmax',
               25: 'swish', 26: 'gelu', 27: 'selu', 28: 'hard_sigmoid', 29: 'softplus',
               30: 'softsign', 31: 'elu', 32: 'hard_tanh'}


def definitions():
    tasks = [
        (3, 'batched_matmul', 'kb_matmul', [(2, 16, 32), (2, 32, 24)], {}, []),
        (4, 'gemv', 'kb_matmul', [(16, 32), (32, 1)], {}, []),
        (5, 'scalar_mul', 'scalar_mul', [(16, 24)], {'scale': 3.14}, []),
        (12, 'diagonal_matmul', 'kb_diagonal', [(16,), (16, 24)], {}, []),
        (16, 'transposed_a_matmul', 'kb_matmul', [(32, 16), (32, 24)], {'transpose_a': True}, []),
        (17, 'transposed_b_matmul', 'kb_matmul', [(16, 32), (24, 32)], {'transpose_b': True}, []),
    ]
    tasks += [(number, name, 'kb_activation', [(4, 65)], {'kind': name}, [1.] if number == 31 else [])
              for number, name in ACTIVATIONS.items()]
    tasks += [(36, 'rms_norm', 'kb_rms_norm', [(2, 8, 5, 7)], {'epsilon': 1e-5}, [8])]
    for number, kind, dims, kernel, stride, padding, dilation in [
        (41, 'max', (2, 3, 65), 8, 1, 4, 3), (42, 'max', (1, 2, 24, 24), 4, 1, 1, 1),
        (44, 'average', (2, 3, 65), 8, 1, 4, 1), (45, 'average', (1, 2, 24, 24), 11, 11, 0, 1),
    ]:
        init = [kernel, stride, padding, dilation, False] if number == 41 else (
            [kernel, stride, padding, dilation] if number == 42 else [kernel, stride, padding])
        tasks.append((number, f'{kind}_pool_{len(dims)-2}d', 'kb_pool', [dims],
                      dict(kind=kind, kernel=kernel, stride=stride, padding=padding, dilation=dilation), init))
    tasks += [(number, kind + '_reduction', 'kb_reduce', [(4, 17, 9)], {'kind': kind, 'axis': 1}, [1])
              for number, kind in [(48, 'mean'), (49, 'max'), (53, 'min')]]
    tasks += [(number, 'reverse_cumsum' if reverse else 'cumsum', 'kb_cumsum', [(4, 65)],
               {'axis': 1, 'reverse': reverse}, [1]) for number, reverse in [(89, False), (91, True)]]
    tasks += [(94, 'mse_loss', 'kb_mse', [(4, 65), (4, 65)], {}, [])]
    return sorted(tasks)


def cuda_body(op, shapes, p, out_shape):
    size = int(np.prod(out_shape))
    if op == 'scalar_mul':
        body = f'out[i] = x[i] * {p["scale"]}f;'
    elif op == 'kb_matmul':
        m, n = out_shape[-2:]
        k = shapes[0][-2] if p.get('transpose_a') else shapes[0][-1]
        ai = f't*{m}+r' if p.get('transpose_a') else f'r*{k}+t'
        bi = f'c*{k}+t' if p.get('transpose_b') else f't*{n}+c'
        body = (f'int batch=i/{m*n}, r=(i/{n})%{m}, c=i%{n}; float s=0; '
                f'for(int t=0;t<{k};++t) s+=x[batch*{m*k}+({ai})]*b[batch*{k*n}+({bi})]; out[i]=s;')
    elif op == 'kb_diagonal':
        body = f'out[i]=x[i/{shapes[1][1]}]*b[i];'
    elif op == 'kb_activation':
        expressions = {
            'relu': 'fmaxf(v,0.0f)', 'leaky_relu': 'v>=0?v:0.01f*v',
            'sigmoid': 'v>=0?1.0f/(1.0f+expf(-v)):expf(v)/(1.0f+expf(v))',
            'tanh': 'tanhf(v)', 'swish': 'v*(v>=0?1.0f/(1.0f+expf(-v)):expf(v)/(1.0f+expf(v)))',
            'gelu': '0.5f*v*(1.0f+erff(v*0.7071067811865475f))',
            'selu': '1.0507009873554805f*(v>0?v:1.6732632423543772f*expm1f(v))',
            'hard_sigmoid': 'fminf(1.0f,fmaxf(0.0f,v/6.0f+0.5f))',
            'softplus': 'v>20?v:log1pf(expf(v))', 'softsign': 'v/(1.0f+fabsf(v))',
            'elu': 'v>0?v:expm1f(v)', 'hard_tanh': 'fminf(1.0f,fmaxf(-1.0f,v))',
        }
        if p['kind'] == 'log_softmax':
            n = shapes[0][1]
            body = (f'int base=(i/{n})*{n}; float mx=x[base],s=0; '
                    f'for(int t=1;t<{n};++t) mx=fmaxf(mx,x[base+t]); '
                    f'for(int t=0;t<{n};++t) s+=expf(x[base+t]-mx); out[i]=(x[i]-mx)-logf(s);')
        else:
            body = 'float v=x[i]; out[i]=' + expressions[p['kind']] + ';'
    elif op == 'kb_rms_norm':
        features, inner = shapes[0][1], int(np.prod(shapes[0][2:]))
        body = (f'int base=(i/{features*inner})*{features*inner}+i%{inner}; float s=0; '
                f'for(int c=0;c<{features};++c) {{float v=x[base+c*{inner}];s+=v*v;}} '
                f'out[i]=x[i]/sqrtf(s/{features}+{p["epsilon"]}f);')
    elif op == 'kb_pool':
        spatial, out_spatial = shapes[0][2:], out_shape[2:]
        k, stride, pad, dilation = (p[key] for key in ('kernel', 'stride', 'padding', 'dilation'))
        initial = '-3.402823466e38f' if p['kind'] == 'max' else '0'
        combine = 's=fmaxf(s,v);' if p['kind'] == 'max' else 's+=v;'
        if len(spatial) == 1:
            body = (f'int group=i/{out_spatial[0]}, a=i%{out_spatial[0]}; float s={initial}; '
                    f'for(int u=0;u<{k};++u) {{int pos=a*{stride}-{pad}+u*{dilation}; '
                    f'if(pos>=0 && pos<{spatial[0]}) {{float v=x[group*{spatial[0]}+pos];{combine}}}}}')
        else:
            h, w = spatial
            oh, ow = out_spatial
            body = (f'int group=i/{oh*ow}, a=(i/{ow})%{oh}, z=i%{ow}; float s={initial}; '
                    f'for(int u=0;u<{k};++u) for(int v=0;v<{k};++v) {{'
                    f'int r=a*{stride}-{pad}+u*{dilation}, c=z*{stride}-{pad}+v*{dilation}; '
                    f'if(r>=0 && r<{h} && c>=0 && c<{w}) {{float v=x[group*{h*w}+r*{w}+c];{combine}}}}}')
        body += 'out[i]=s' + (f'/{k**len(spatial)}.0f;' if p['kind'] == 'average' else ';')
    elif op == 'kb_reduce':
        _, axis, inner = shapes[0]
        initial = '0' if p['kind'] == 'mean' else f'x[(i/{inner})*{axis*inner}+i%{inner}]'
        expression = {'mean': 's+=v;', 'max': 's=fmaxf(s,v);', 'min': 's=fminf(s,v);'}[p['kind']]
        body = (f'int base=(i/{inner})*{axis*inner}+i%{inner}; float s={initial}; '
                f'for(int t=0;t<{axis};++t) {{float v=x[base+t*{inner}];{expression}}} '
                'out[i]=s' + (f'/{axis}.0f;' if p['kind'] == 'mean' else ';'))
    elif op == 'kb_cumsum':
        n = shapes[0][1]
        bounds = f'i%{n};t<{n}' if p['reverse'] else f'0;t<=i%{n}'
        body = f'float s=0;for(int t={bounds};++t) s+=x[(i/{n})*{n}+t];out[i]=s;'
    elif op == 'kb_mse':
        n = int(np.prod(shapes[0]))
        body = f'float s=0;for(int t=0;t<{n};++t) {{float d=x[t]-b[t];s+=d*d;}}out[i]=s/{n}.0f;'
    else:
        raise ValueError(op)
    arguments = 'const float *x, ' + ('const float *b, ' if len(shapes) == 2 else '') + 'float *out'
    return ('extern "C" __global__ void kernel(' + arguments + ') {\n'
            f'    int i=blockIdx.x*blockDim.x+threadIdx.x;\n    if(i<{size}) {{ {body} }}\n}}\n')


def prepare(destination):
    destination.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(f'https://api.github.com/repos/ScalingIntelligence/KernelBench/contents/KernelBench/level1?ref={COMMIT}', timeout=30) as response:
        names = {int(row['name'].split('_')[0]): row['name'] for row in json.load(response) if row['name'].endswith('.py')}
    with urllib.request.urlopen(BASE + '/LICENSE', timeout=30) as response:
        (destination / 'LICENSE.KernelBench').write_bytes(response.read())
    for number, name, op, shapes, params, init in definitions():
        folder = destination / f'{number:03d}_{name}'
        folder.mkdir()
        upstream_url = BASE + '/KernelBench/level1/' + names[number]
        with urllib.request.urlopen(upstream_url, timeout=30) as response:
            upstream = response.read()
        (folder / 'upstream.py').write_bytes(upstream)
        arrays = [np.ones(shape, dtype=np.float32) for shape in shapes]
        y = arrays[0] * params['scale'] if op == 'scalar_mul' else reference(op, arrays, params)
        out_shape = y.shape
        tensors = [dict(name='x' if index == 0 else 'b', direction='input', dtype='float32', shape=list(shape))
                   for index, shape in enumerate(shapes)]
        tensors.append(dict(name='out', direction='output', dtype='float32', shape=list(out_shape)))
        manifest = dict(name=f'kb_{number:03d}_{name}', source_file='kernel.cu', dialect='cuda', entrypoint='kernel',
                        tensors=tensors, constraints=['Static contiguous row-major tensors; preserve the source mathematics.'],
                        tolerance={'rtol': 1e-5, 'atol': 1e-6}, seed=20261001 + number,
                        oracle={'operation': op, 'parameters': params},
                        provenance={'repository': 'https://github.com/ScalingIntelligence/KernelBench', 'commit': COMMIT,
                                    'upstream_file': names[number], 'upstream_url': upstream_url,
                                    'upstream_sha256': hashlib.sha256(upstream).hexdigest(),
                                    'upstream_init': json.dumps(init), 'license': 'MIT',
                                    'adaptation': 'float32 CUDA; reduced static shapes; scalar result reshaped to [1]; operator defaults preserved'})
        manifest = KernelManifest.model_validate(manifest)
        (folder / 'manifest.json').write_text(manifest.model_dump_json(indent=2) + '\n')
        (folder / 'kernel.cu').write_text(cuda_body(op, shapes, params, out_shape))
    (destination / 'README.md').write_text(
        '# KernelBench CUDA adaptations\n\nThirty distinct Level-1 tasks pinned to `' + COMMIT + '`.\n'
        'The upstream MIT sources are retained beside each adaptation. Static float32 shapes are reduced for\n'
        'offline correctness evaluation; these are not original KernelBench performance results. Manifests\n'
        'record source provenance and frozen tolerances. Use scripts/validate_kernelbench_sources.py to\n'
        'compare CUDA, upstream PyTorch, and the independent NumPy references on all twelve input families.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('examples/kernelbench'))
    prepare(parser.parse_args().output)
