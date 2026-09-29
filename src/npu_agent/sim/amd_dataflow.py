from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import random
import time
from typing import Callable

import numpy as np

from .amd_mlir_frontend import DataflowProgram, UnsupportedMLIR


class SimulationError(ValueError):
    pass


@dataclass
class Slot:
    generation: int
    data: np.ndarray
    published: bool = False
    released: set[str] = field(default_factory=set)


class Fifo:
    def __init__(self, spec: dict):
        if not spec["consumers"] or len(set(spec["consumers"])) != len(spec["consumers"]) or spec["producer"] in spec["consumers"]:
            raise SimulationError("unsupported or duplicate FIFO endpoints")
        self.spec = spec
        self.slots: list[Slot] = []
        self.held: dict[str, list[Slot]] = {}
        self.generation = 0

    def acquire(self, actor: str, port: str, count: int) -> list[Slot] | None:
        if count <= 0 or count > self.spec['capacity']:
            raise SimulationError(f'invalid acquire count {count} for capacity {self.spec["capacity"]}')
        expected = [self.spec['producer']] if port == 'produce' else self.spec['consumers']
        if actor not in expected:
            raise SimulationError(f'wrong FIFO endpoint: {actor}, {port}')
        held = self.held.setdefault(actor, [])
        needed = max(0, count - len(held))
        if port == 'produce':
            if len(self.slots) + needed > self.spec['capacity']:
                return None
            for _ in range(needed):
                dtype = np.dtype(self.spec['dtype'])
                data = np.full(self.spec['shape'], np.nan if np.issubdtype(dtype, np.floating) else np.iinfo(dtype).min, dtype=dtype)
                slot = Slot(self.generation, data)
                self.generation += 1
                self.slots.append(slot)
                held.append(slot)
        else:
            held_ids = {s.generation for s in held}
            available = [s for s in self.slots if s.published and actor not in s.released and s.generation not in held_ids]
            if len(available) < needed:
                return None
            held.extend(available[:needed])
        # Acquire(N) means an N-object window, including already held objects.
        return held[:count]

    def release(self, actor: str, port: str, count: int):
        held = self.held.get(actor, [])
        if count <= 0 or count > len(held):
            raise SimulationError(f'double or excessive release by {actor}')
        for slot in held[:count]:
            if port == 'produce':
                if slot.published:
                    raise SimulationError('double publication')
                slot.published = True
            else:
                slot.released.add(actor)
        del held[:count]
        self.slots[:] = [s for s in self.slots if not (s.published and set(self.spec['consumers']) <= s.released)]

    def owns(self, actor: str, slot: Slot) -> bool:
        return any(s is slot for s in self.held.get(actor, []))


def simulate(program: DataflowProgram, inputs: dict[str, np.ndarray], tensor_contracts: list[dict],
             kernel: Callable, seed: int = 0, max_steps: int = 100000, timeout_seconds: float = 60) -> tuple[dict, dict]:
    started = time.monotonic()
    if len(program.arguments) != len(tensor_contracts):
        raise SimulationError('runtime ABI argument count mismatch')
    host = []
    written = []
    for abi, tensor in zip(program.arguments, tensor_contracts, strict=True):
        if abi['shape'] != tensor['shape'] or abi['dtype'] != tensor['dtype']:
            raise SimulationError('runtime ABI shape/dtype mismatch')
        host.append(inputs[tensor['name']].copy() if tensor['direction'] == 'input' else np.zeros(tensor['shape'], tensor['dtype']))
        written.append(np.zeros(tensor['shape'], dtype=bool))
    fifos = {name: Fifo(spec) for name, spec in program.fifos.items()}
    pcs = {name: 0 for name in program.workers}
    env = {name: {} for name in program.workers}
    transfers = {}
    for name, spec in program.transfers.items():
        if spec['fifo'] not in fifos:
            raise SimulationError('unknown DMA FIFO')
        fifo = fifos[spec['fifo']]
        index = spec['argument']
        if index < 0 or index >= len(host):
            raise SimulationError('DMA argument out of range')
        shape = spec['sizes']
        if len(shape) != len(spec['strides']) or any(s <= 0 for s in shape):
            raise SimulationError('invalid DMA sizes')
        if int(np.prod(shape, dtype=object)) != spec['length']:
            raise SimulationError('DMA length disagrees with sizes')
        indices = [spec['offset'] + sum(i * stride for i, stride in zip(pos, spec['strides'], strict=True))
                   for pos in itertools.product(*(range(s) for s in shape))]
        if not indices or min(indices) < 0 or max(indices) >= host[index].size:
            raise SimulationError('DMA bounds violation')
        size = int(np.prod(fifo.spec['shape']))
        if len(indices) % size or host[index].dtype != np.dtype(fifo.spec['dtype']):
            raise SimulationError('DMA token count or element size mismatch')
        direction = tensor_contracts[index]['direction']
        actor = fifo.spec['producer'] if direction == 'input' else next((a for a in fifo.spec['consumers'] if a not in program.workers), None)
        if actor is None or actor in program.workers:
            raise SimulationError('DMA endpoint is not a host tile')
        transfers[name] = {'spec': spec, 'indices': indices, 'position': 0, 'active': False,
                           'done': False, 'freed': False, 'actor': actor, 'direction': direction}
    links = {}
    for index, link in enumerate(program.links):
        ins, outs = link['inputs'], link['outputs']
        if not ins or not outs or (len(ins) != 1 and len(outs) != 1):
            raise SimulationError('only one-to-one, split, or join links are supported')
        sources = [fifos[n] for n in ins]
        destinations = [fifos[n] for n in outs]
        actor = destinations[0].spec['producer']
        if actor in program.workers or any(d.spec['producer'] != actor for d in destinations) or any(actor not in f.spec['consumers'] for f in sources):
            raise SimulationError('invalid link tile')
        if len({f.spec['dtype'] for f in sources + destinations}) != 1:
            raise SimulationError('link element type mismatch')
        sizes_in = [int(np.prod(f.spec['shape'])) for f in sources]
        sizes_out = [int(np.prod(f.spec['shape'])) for f in destinations]
        if len(ins) == 1:
            offsets = link['dst_offsets'] or [0]
            limit, sizes = sizes_in[0], sizes_out
        else:
            offsets = link['src_offsets']
            limit, sizes = sizes_out[0], sizes_in
        if len(offsets) != len(sizes):
            raise SimulationError('link offset count mismatch')
        coverage = np.zeros(limit, dtype=int)
        for offset, size in zip(offsets, sizes, strict=True):
            if offset < 0 or offset + size > limit:
                raise SimulationError('link bounds violation')
            coverage[offset:offset + size] += 1
        if not np.all(coverage == 1):
            raise SimulationError('link must cover each element exactly once')
        links[f'link_{index}'] = {'actor': actor, 'sources': sources, 'destinations': destinations, 'offsets': offsets}
    runtime_pc = 0
    rng = random.Random(seed)
    wait_graph = {}
    steps = 0
    while True:
        if steps >= max_steps or time.monotonic() - started > timeout_seconds:
            raise SimulationError('simulation_step_or_time_bound: inconclusive')
        actors = ['runtime', *program.workers, *transfers, *links]
        rng.shuffle(actors)
        progressed = False
        wait_graph = {}
        for actor in actors:
            steps += 1
            if actor == 'runtime':
                if runtime_pc >= len(program.runtime):
                    continue
                op = program.runtime[runtime_pc]
                task = transfers[op['task']]
                if op['op'] == 'start':
                    if task['active'] or task['done'] or task['freed']:
                        raise SimulationError('DMA task reused')
                    task['active'] = True
                elif op['op'] in ('await', 'free'):
                    if op['op'] == 'await' and not task['spec']['issue_token']:
                        raise SimulationError('await on DMA without completion token')
                    if not task['done']:
                        wait_graph[actor] = {'waits_for': op['task']}
                        continue
                    if op['op'] == 'free':
                        if task['freed']:
                            raise SimulationError('double DMA free')
                        task['freed'] = True
                else:
                    raise SimulationError('unknown runtime instruction')
                runtime_pc += 1
            elif actor in links:
                link = links[actor]
                owner = link['actor']
                input_windows = [f.acquire(owner, 'consume', 1) for f in link['sources']]
                # Links have no independent completion token; idle links are done
                # when runtime transfers and finite workers finish with no held slots.
                if any(w is None for w in input_windows):
                    continue
                output_windows = [f.acquire(owner, 'produce', 1) for f in link['destinations']]
                if any(w is None for w in output_windows):
                    wait_graph[actor] = {'waits_for': 'link destination slots'}
                    continue
                if len(input_windows) == 1:
                    data = input_windows[0][0].data.reshape(-1)
                    for window, offset in zip(output_windows, link['offsets'], strict=True):
                        window[0].data.flat[:] = data[offset:offset + window[0].data.size]
                else:
                    data = output_windows[0][0].data.reshape(-1)
                    for window, offset in zip(input_windows, link['offsets'], strict=True):
                        data[offset:offset + window[0].data.size] = window[0].data.reshape(-1)
                for fifo in link['sources']:
                    fifo.release(owner, 'consume', 1)
                for fifo in link['destinations']:
                    fifo.release(owner, 'produce', 1)
            elif actor in transfers:
                task = transfers[actor]
                if not task['active'] or task['done']:
                    continue
                fifo = fifos[task['spec']['fifo']]
                port = 'produce' if task['direction'] == 'input' else 'consume'
                window = fifo.acquire(task['actor'], port, 1)
                if window is None:
                    wait_graph[actor] = {'fifo': task['spec']['fifo'], 'port': port, 'count': 1}
                    continue
                slot = window[0]
                pos, size = task['position'], slot.data.size
                indices = task['indices'][pos:pos + size]
                array = host[task['spec']['argument']].reshape(-1)
                if port == 'produce':
                    slot.data.flat[:] = array[indices]
                else:
                    array[indices] = slot.data.reshape(-1)
                    written[task['spec']['argument']].flat[indices] = True
                fifo.release(task['actor'], port, 1)
                task['position'] += size
                task['done'] = task['position'] == len(task['indices'])
            else:
                pc = pcs[actor]
                if pc >= len(program.workers[actor]):
                    continue
                instruction = program.workers[actor][pc]
                op = instruction['op']
                local = env[actor]
                if op in ('acquire', 'release'):
                    fifo = fifos[instruction['fifo']]
                    if op == 'acquire':
                        window = fifo.acquire(actor, instruction['port'], instruction['count'])
                        if window is None:
                            wait_graph[actor] = {**instruction, 'producer': fifo.spec['producer'], 'consumers': fifo.spec['consumers']}
                            continue
                        local[instruction['result']] = (fifo, window, instruction['port'])
                    else:
                        fifo.release(actor, instruction['port'], instruction['count'])
                elif op == 'access':
                    fifo, window, port = local[instruction['window']]
                    if not 0 <= instruction['index'] < len(window):
                        raise SimulationError('FIFO subview index out of bounds')
                    slot = window[instruction['index']]
                    if not fifo.owns(actor, slot):
                        raise SimulationError('use after release')
                    local[instruction['result']] = (fifo, slot, port)
                elif op == 'call':
                    args, readonly = [], []
                    for argument in instruction['args']:
                        if 'constant' in argument:
                            args.append(argument['constant'])
                            readonly.append(True)
                        else:
                            fifo, slot, port = local[argument['view']]
                            if not fifo.owns(actor, slot):
                                raise SimulationError('use after release at kernel call')
                            if any(value is slot.data for value in args):
                                raise UnsupportedMLIR('aliased kernel arguments require a shared-buffer host ABI')
                            args.append(slot.data)
                            readonly.append(port == 'consume')
                    kernel(instruction['kernel'], args, readonly)
                else:
                    raise SimulationError(f'unknown worker instruction: {op}')
                pcs[actor] += 1
            progressed = True
        complete = (runtime_pc == len(program.runtime) and all(pcs[a] == len(program.workers[a]) for a in pcs)
                    and all(t['done'] and t['freed'] for t in transfers.values()))
        if complete:
            if any(f.slots or any(f.held.values()) for f in fifos.values()):
                raise SimulationError('unreleased FIFO objects or excess token counts')
            outputs = {}
            for i, tensor in enumerate(tensor_contracts):
                if tensor['direction'] == 'output':
                    if not written[i].all():
                        raise SimulationError('incomplete output DMA coverage')
                    outputs[tensor['name']] = host[i]
            return outputs, {'logical_steps': steps, 'schedule_seed': seed, 'duration_seconds': time.monotonic() - started}
        if not progressed:
            raise SimulationError(f'simulation_deadlock: {wait_graph}')
