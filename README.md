# graph_bench

A microbenchmark for measuring `hipGraphLaunch` submission latency across
different graph topologies and sizes on AMD GPUs.

## Motivation

HIP graphs allow pre-recording sequences of GPU operations and replaying them
with a single API call. The submission overhead depends not just on graph size
(number of nodes) but also on graph *shape* — how many independent chains of
nodes exist and whether those chains can run concurrently on the GPU.

This benchmark makes it easy to:

- Measure submission-only latency (`--no-sync`) to isolate CPU-side overhead.
- Measure end-to-end latency (`--sync`) to see the effect of GPU parallelism.
- Compare multiple graph topologies side-by-side for the same total node count.

## Topologies

All topologies use approximately **N** total kernel nodes.

| Name       | Description |
|------------|-------------|
| `straight` | Single linear chain of N nodes (1 segment). Baseline. |
| `paths2`   | Lead chain (N/4) → 2 parallel branches (N/4 each) → tail chain (N/4). 4 segments total. |
| `paths4`   | Lead chain (N/6) → 4 parallel branches (N/6 each) → tail chain (N/6). 6 segments total. |
| `full2`    | 2 fully independent chains of N/2 nodes each. No sync point. 2 segments. |
| `full4`    | 4 fully independent chains of N/4 nodes each. No sync point. 4 segments. |

A *segment* is a maximal contiguous chain of nodes with no cross-dependencies.
Each segment maps to one AQL packet batch submission, so fewer segments means
lower submission overhead.

## Build

The benchmark is written against the HIP API. When compiled with `nvcc` the
`hip*` symbols are automatically remapped to their `cuda*` equivalents via
`#define` aliases at the top of the source file — no source changes needed.

### HIP / AMD (default)

```bash
# With hipcc directly
/opt/rocm/bin/hipcc -O2 -o graph_bench graph_bench.cpp

# With CMake
cmake -B build -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build
```

### CUDA / NVIDIA

```bash
# With nvcc directly
nvcc -O2 -x cu -o graph_bench graph_bench.cpp

# With CMake
cmake -B build -DUSE_CUDA=ON
cmake --build build
```

## Usage

```
./graph_bench [options]

Options:
  --size N            Total number of kernel nodes (default: 1024)
  --graphSize N       Alias for --size
  --iters N           Timed repetitions per measurement (default: 1000)
  --no-sync           Measure submission latency only (default)
  --sync              Measure submission + GPU execution latency
  --sweep             Run across all sizes: 1, 2, 4, 8, ..., 8192
  --topology <name>   Benchmark only the named topology (default: all)
  --instantiate       Measure hipGraphInstantiate time (alongside launch)
  --verify            Run ordering correctness check instead of timing
  --verify-iters N    Verify launches per topology (default: 50)
  --verify-delay-us N Per-node busy-wait to widen the race window (default: 1)
```

## Examples

Run all topologies at size 1024, submission only:

```bash
./graph_bench --size 1024 --no-sync
```

Run only `full4` with GPU execution included:

```bash
./graph_bench --topology full4 --size 1024 --sync
```

Sweep all sizes for all topologies:

```bash
./graph_bench --sweep --no-sync
```

Sweep a single topology across all sizes:

```bash
./graph_bench --sweep --topology paths2
```

Measure instantiation time alongside launch latency:

```bash
./graph_bench --instantiate --size 1024
```

Sweep with instantiation for a specific topology:

```bash
./graph_bench --sweep --topology straight --instantiate
```

Verify ordering correctness (all topologies):

```bash
./graph_bench --verify --size 1024
```

Stress a single topology harder for race detection:

```bash
./graph_bench --verify --topology paths4 --verify-iters 200 --verify-delay-us 10
```

## Verification

`--verify` swaps the empty timing kernel for a reduction kernel that computes
`buf[nodeId] = sum(buf[dep_ids]) + 1`. Expected values are computed on the CPU
at build time using the same recurrence, and after each launch every node (not
just the exits) is compared against its expected value. A node that ran before
one of its dependencies reads a stale `0`, producing a deficit that is caught
by the comparison.

To make nondeterministic ordering bugs observable, verification:

- launches each graph `--verify-iters` times (default 50), resetting the buffer
  before every launch, since a single launch can pass by luck; and
- has every node read its dependencies *early* and write its own result *late*,
  after busy-waiting `--verify-delay-us` microseconds (default 1). This widens
  the window during which an out-of-order successor would observe the stale `0`.

Increase both knobs to hunt harder for dependency-handling bugs in the runtime.

## Sample Output

### Launch sweep (default)

```
Device : AMD Instinct MI300X
Mode   : no-sync (submission only)
Iters  : 1000 per measurement

--- launch (us) ---
size      straight      paths2      paths4       full2       full4
---------------------------------------------------------------------
1            0.798       0.912       1.102       0.854       1.023
...
1024        15.231      18.442      22.105      16.312      20.877
8192       120.154     143.211     178.334     127.442     159.221
```

### Instantiate + launch (--instantiate)

```
Device : AMD Instinct MI300X
Mode   : no-sync (submission only)
Iters  : 1000 per measurement
Metrics: instantiate + launch

topology      inst (us)    launch (us)
--------------------------------------
straight        45.678         15.231
paths2          78.901         18.442
paths4         112.345         22.105
full2           56.789         16.312
full4           89.012         20.877
```

### Sweep with topology filter (--sweep --topology paths2)

```
Device : AMD Instinct MI300X
Mode   : no-sync (submission only)
Iters  : 1000 per measurement

--- launch (us) ---
size        paths2
-------------------
1            0.912
...
1024        18.442
8192       143.211
```

### Verification (--verify)

```
Device : AMD Instinct MI300X
Mode   : verify (ordering check, size=1024, iters=50, delay=1us/node)

topology    exits   result
--------------------------------
straight    1       PASS
paths2      1       PASS
paths4      1       PASS
full2       2       PASS
full4       4       PASS
```

## License

MIT — Copyright (c) 2026 Saleel Kudchadker
