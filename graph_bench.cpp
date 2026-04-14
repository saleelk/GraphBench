// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Saleel Kudchadker
//
// graph_bench.cpp
//
// Measures hipGraphLaunch latency across several graph topologies for a
// sweep of graph sizes.  All topologies use approximately N total kernel
// nodes so that the comparison is apples-to-apples.
//
// Topologies
// ----------
//   straight  1 segment of N nodes chained linearly.
//
//   paths2    Hexagon-style with 2 parallel branches:
//               lead (N/4) -> branch_0 (N/4) + branch_1 (N/4) -> tail (N/4)
//             Produces 4 segments.
//
//   paths4    Same pattern with 4 parallel branches, each of length N/6:
//               lead (N/6) -> branch_{0..3} (N/6 each) -> tail (N/6)
//             Produces 6 segments.
//
//   full2     2 fully independent chains of N/2 nodes each.
//             No synchronisation point between them — 2 segments.
//
//   full4     4 fully independent chains of N/4 nodes each — 4 segments.
//
// Build:
//   /opt/rocm/bin/hipcc -O2 -o graph_bench graph_bench.cpp
//
// Usage:
//   ./graph_bench [--size N] [--iters N] [--no-sync] [--sync]
//                 [--sweep] [--topology <name>]
//
//   --size N          Graph size / node count (default: 1024)
//   --graphSize N     Alias for --size
//   --iters N         Timed repetitions per measurement (default: 1000)
//   --no-sync         Submission latency only (default)
//   --sync            Submission + GPU execution latency
//   --sweep           Run all sizes from 1 to 8192
//   --topology <name> Benchmark only the named topology (default: all)

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", _e,                     \
              hipGetErrorString(_e), __FILE__, __LINE__);                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

class Timer {
 public:
  void reserve(int n) { samples_.reserve(n); }

  void start() { t_ = std::chrono::high_resolution_clock::now(); }

  void stop() {
    samples_.push_back(std::chrono::duration<double, std::micro>(
                           std::chrono::high_resolution_clock::now() - t_)
                           .count());
  }

  double avg() const {
    return std::accumulate(samples_.begin(), samples_.end(), 0.0) /
           samples_.size();
  }
  double min() const {
    return *std::min_element(samples_.begin(), samples_.end());
  }
  double max() const {
    return *std::max_element(samples_.begin(), samples_.end());
  }

 private:
  std::chrono::high_resolution_clock::time_point t_;
  std::vector<double> samples_;
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void null_kernel() {}

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

// straight: single linear chain of N nodes.
static hipGraphExec_t build_straight(int N) {
  hipGraph_t g;
  HIP_CHECK(hipGraphCreate(&g, 0));

  hipKernelNodeParams p{};
  p.func     = reinterpret_cast<void*>(null_kernel);
  p.gridDim  = {1, 1, 1};
  p.blockDim = {1, 1, 1};

  hipGraphNode_t prev{}, cur{};
  for (int i = 0; i < N; ++i) {
    HIP_CHECK(hipGraphAddKernelNode(&cur, g, i == 0 ? nullptr : &prev,
                                    i == 0 ? 0 : 1, &p));
    prev = cur;
  }

  hipGraphExec_t e;
  HIP_CHECK(hipGraphInstantiate(&e, g, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(g));
  return e;
}

// multi-path (hexagon): lead -> P parallel branches -> tail.
//   seg = N / (P + 2)  nodes per segment
//   total segments = P + 2
static hipGraphExec_t build_multi_path(int N, int P) {
  const int seg = std::max(1, N / (P + 2));

  hipGraph_t g;
  HIP_CHECK(hipGraphCreate(&g, 0));

  hipKernelNodeParams p{};
  p.func     = reinterpret_cast<void*>(null_kernel);
  p.gridDim  = {1, 1, 1};
  p.blockDim = {1, 1, 1};

  // Leading straight chain.
  hipGraphNode_t prev{}, cur{};
  for (int i = 0; i < seg; ++i) {
    HIP_CHECK(hipGraphAddKernelNode(&cur, g, i == 0 ? nullptr : &prev,
                                    i == 0 ? 0 : 1, &p));
    prev = cur;
  }
  hipGraphNode_t split_end = prev;

  // P parallel branches, each depending only on split_end.
  std::vector<hipGraphNode_t> path_ends(P);
  for (int path = 0; path < P; ++path) {
    hipGraphNode_t pprev = split_end, pcur{};
    for (int i = 0; i < seg; ++i) {
      HIP_CHECK(hipGraphAddKernelNode(&pcur, g, &pprev, 1, &p));
      pprev = pcur;
    }
    path_ends[path] = pprev;
  }

  // Join node that waits for all branches, then trailing chain.
  hipGraphNode_t join{};
  HIP_CHECK(hipGraphAddKernelNode(&join, g, path_ends.data(), P, &p));
  prev = join;
  for (int i = 1; i < seg; ++i) {
    HIP_CHECK(hipGraphAddKernelNode(&cur, g, &prev, 1, &p));
    prev = cur;
  }

  hipGraphExec_t e;
  HIP_CHECK(hipGraphInstantiate(&e, g, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(g));
  return e;
}

static hipGraphExec_t build_paths2(int N) { return build_multi_path(N, 2); }
static hipGraphExec_t build_paths4(int N) { return build_multi_path(N, 4); }

// fully parallel: P independent chains of N/P nodes.
// No synchronisation point — GPU can schedule all chains concurrently.
static hipGraphExec_t build_full_parallel(int N, int P) {
  const int seg = std::max(1, N / P);

  hipGraph_t g;
  HIP_CHECK(hipGraphCreate(&g, 0));

  hipKernelNodeParams p{};
  p.func     = reinterpret_cast<void*>(null_kernel);
  p.gridDim  = {1, 1, 1};
  p.blockDim = {1, 1, 1};

  for (int path = 0; path < P; ++path) {
    hipGraphNode_t pprev{}, pcur{};
    for (int i = 0; i < seg; ++i) {
      HIP_CHECK(hipGraphAddKernelNode(&pcur, g, i == 0 ? nullptr : &pprev,
                                      i == 0 ? 0 : 1, &p));
      pprev = pcur;
    }
  }

  hipGraphExec_t e;
  HIP_CHECK(hipGraphInstantiate(&e, g, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(g));
  return e;
}

static hipGraphExec_t build_full2(int N) { return build_full_parallel(N, 2); }
static hipGraphExec_t build_full4(int N) { return build_full_parallel(N, 4); }

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

// Returns average hipGraphLaunch time in microseconds over `iters` launches.
// If syncInTiming is true, hipStreamSynchronize is included in each sample.
// Between samples the stream is always drained to avoid carry-over GPU work.
static double bench(hipGraphExec_t exec, int iters, bool syncInTiming) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Warm-up.
  for (int i = 0; i < 10; ++i) HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  Timer t;
  t.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    t.start();
    HIP_CHECK(hipGraphLaunch(exec, stream));
    if (syncInTiming) HIP_CHECK(hipStreamSynchronize(stream));
    t.stop();
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  HIP_CHECK(hipStreamDestroy(stream));
  return t.avg();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  int         size          = 1024;
  int         iters         = 1000;
  bool        syncInTiming  = false;
  bool        sweep         = false;
  std::string topo          = "all";

  for (int i = 1; i < argc; ++i) {
    if ((!strcmp(argv[i], "--size") || !strcmp(argv[i], "--graphSize")) &&
        i + 1 < argc)
      size = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--no-sync"))
      syncInTiming = false;
    else if (!strcmp(argv[i], "--sync"))
      syncInTiming = true;
    else if (!strcmp(argv[i], "--sweep"))
      sweep = true;
    else if (!strcmp(argv[i], "--topology") && i + 1 < argc)
      topo = argv[++i];
  }

  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  printf("Device : %s\n", props.name);
  printf("Mode   : %s\n",
         syncInTiming ? "sync (submission+GPU)" : "no-sync (submission only)");
  printf("Iters  : %d per measurement\n\n", iters);

  struct Topo {
    const char*       name;
    hipGraphExec_t  (*build)(int);
  };

  const Topo topos[] = {
      {"straight", build_straight},
      {"paths2",   build_paths2  },
      {"paths4",   build_paths4  },
      {"full2",    build_full2   },
      {"full4",    build_full4   },
  };
  const int ntopos = static_cast<int>(sizeof(topos) / sizeof(topos[0]));

  if (sweep) {
    const int sweep_sizes[] = {1,   2,   4,    8,    16,   32,
                                64,  128, 256,  512,  1024, 2048,
                                4096, 8192};
    const int nsizes =
        static_cast<int>(sizeof(sweep_sizes) / sizeof(sweep_sizes[0]));

    // Header row.
    printf("%-7s", "size");
    for (int t = 0; t < ntopos; ++t) printf("  %10s", topos[t].name);
    printf("\n%s\n", std::string(7 + ntopos * 12, '-').c_str());

    for (int s = 0; s < nsizes; ++s) {
      const int N = sweep_sizes[s];
      printf("%-7d", N);
      for (int t = 0; t < ntopos; ++t) {
        hipGraphExec_t e = topos[t].build(N);
        const double avg = bench(e, iters, syncInTiming);
        HIP_CHECK(hipGraphExecDestroy(e));
        printf("  %9.3f us", avg);
      }
      printf("\n");
      fflush(stdout);
    }
  } else {
    printf("%-10s  %s\n", "topology", "avg (us)");
    printf("%s\n", std::string(30, '-').c_str());
    for (int t = 0; t < ntopos; ++t) {
      if (topo != "all" && topo != topos[t].name) continue;
      hipGraphExec_t e = topos[t].build(size);
      const double avg = bench(e, iters, syncInTiming);
      HIP_CHECK(hipGraphExecDestroy(e));
      printf("%-10s  %.3f us\n", topos[t].name, avg);
    }
  }

  return 0;
}
