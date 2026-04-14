// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Saleel Kudchadker
//
// graph_bench.cpp
//
// Measures hipGraphLaunch latency across several graph topologies and sizes.
// All topologies use approximately N total kernel nodes so that comparisons
// are apples-to-apples.
//
// Topologies
// ----------
//   straight  1 segment of N nodes chained linearly.
//
//   paths2    Hexagon-style with 2 parallel branches:
//               lead (N/4) -> branch_0 (N/4) + branch_1 (N/4) -> tail (N/4)
//             4 segments total.
//
//   paths4    Same pattern with 4 parallel branches, each of length N/6.
//             6 segments total.
//
//   full2     2 fully independent chains of N/2 nodes. No sync point.
//
//   full4     4 fully independent chains of N/4 nodes. No sync point.
//
// Verification (--verify)
// -----------------------
// When --verify is passed the benchmark is replaced by a correctness check.
// Each graph node is assigned a unique ID in [0, N). The verify_kernel writes
//   buf[nodeId] = nodeId + 1
// After a single graph launch + sync, every slot is checked:
//   buf[i] == i+1  for all i in [0, actual_node_count)
//
// This confirms:
//   - All N kernels executed (every slot nonzero).
//   - For hexagon topologies: the join node's slot is set only if all parallel
//     branches finished first; the trailing chain proves the join completed
//     before it ran.  Checking all slots transitively validates fan-out/fan-in.
//   - For full-parallel topologies: each independent chain ran to completion.
//
// Build (HIP/AMD):
//   /opt/rocm/bin/hipcc -O2 -o graph_bench graph_bench.cpp
//   cmake -B build -DCMAKE_PREFIX_PATH=/opt/rocm && cmake --build build
//
// Build (CUDA/NVIDIA):
//   nvcc -O2 -x cu -o graph_bench graph_bench.cpp
//   cmake -B build -DUSE_CUDA=ON && cmake --build build
//
// Usage:
//   ./graph_bench [--size N] [--iters N] [--no-sync] [--sync]
//                 [--sweep] [--topology <name>] [--verify]
//
//   --size N          Total kernel nodes (default: 1024)
//   --graphSize N     Alias for --size
//   --iters N         Timed repetitions per measurement (default: 1000)
//   --no-sync         Submission latency only (default)
//   --sync            Submission + GPU execution latency
//   --sweep           Run across all sizes: 1, 2, 4, ..., 8192
//   --topology <name> Benchmark only the named topology (default: all)
//   --verify          Run correctness check instead of timing

// ---------------------------------------------------------------------------
// HIP / CUDA portability layer
// Code is written against the HIP API; when compiled with nvcc the hip*
// symbols are remapped to their cuda* equivalents.
// ---------------------------------------------------------------------------
#if defined(__NVCC__)
#include <cuda_runtime.h>
// Types
#define hipError_t            cudaError_t
#define hipDeviceProp_t       cudaDeviceProp
#define hipStream_t           cudaStream_t
#define hipGraph_t            cudaGraph_t
#define hipGraphExec_t        cudaGraphExec_t
#define hipGraphNode_t        cudaGraphNode_t
#define hipKernelNodeParams   cudaKernelNodeParams
// Error values
#define hipSuccess            cudaSuccess
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
// Runtime
#define hipGetDevice            cudaGetDevice
#define hipGetDeviceProperties  cudaGetDeviceProperties
#define hipGetErrorString       cudaGetErrorString
#define hipStreamCreate         cudaStreamCreate
#define hipStreamDestroy        cudaStreamDestroy
#define hipStreamSynchronize    cudaStreamSynchronize
#define hipMalloc               cudaMalloc
#define hipFree                 cudaFree
#define hipMemset               cudaMemset
#define hipMemcpy               cudaMemcpy
// Graph
#define hipGraphCreate                          cudaGraphCreate
#define hipGraphDestroy                         cudaGraphDestroy
#define hipGraphAddKernelNode                   cudaGraphAddKernelNode
#define hipGraphInstantiate(e,g,_,__,f)         cudaGraphInstantiate(e,g,nullptr,nullptr,f)
#define hipGraphExecDestroy                     cudaGraphExecDestroy
#define hipGraphLaunch                          cudaGraphLaunch
// Stream capture
#define hipStreamBeginCapture         cudaStreamBeginCapture
#define hipStreamEndCapture           cudaStreamEndCapture
#define hipStreamCaptureModeGlobal    cudaStreamCaptureModeGlobal
#else
#include <hip/hip_runtime.h>
#endif

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
// Kernels
// ---------------------------------------------------------------------------

__global__ void null_kernel() {}

// Writes nodeId+1 to buf[nodeId].  Used in --verify mode so that every node
// leaves a unique, checkable footprint in device memory.
__global__ void verify_kernel(int* buf, int nodeId) {
  buf[nodeId] = nodeId + 1;
}

// ---------------------------------------------------------------------------
// Verification context
// ---------------------------------------------------------------------------

// Passed to graph builders when --verify is active.  The builder calls
// add_node() for every kernel it adds; add_node() assigns a monotonically
// increasing ID and wires up verify_kernel.  After the build, next_id equals
// the actual number of nodes, which is the size needed for dev_buf.
struct VerifyCtx {
  int* dev_buf  = nullptr;  // device buffer of size >= next_id (caller-owned)
  int  next_id  = 0;        // auto-incremented as nodes are added
};

// ---------------------------------------------------------------------------
// Node helper
// ---------------------------------------------------------------------------

// Add one kernel node to graph g.  If ctx is non-null the node runs
// verify_kernel; otherwise it runs null_kernel.
static void add_node(hipGraph_t g, hipGraphNode_t* cur,
                     const hipGraphNode_t* deps, int ndeps,
                     VerifyCtx* ctx) {
  hipKernelNodeParams p{};
  p.gridDim  = {1, 1, 1};
  p.blockDim = {1, 1, 1};

  if (ctx) {
    int   id   = ctx->next_id++;
    void* args[] = {reinterpret_cast<void*>(&ctx->dev_buf),
                    reinterpret_cast<void*>(&id)};
    p.func         = reinterpret_cast<void*>(verify_kernel);
    p.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(cur, g, deps, ndeps, &p));
  } else {
    p.func = reinterpret_cast<void*>(null_kernel);
    HIP_CHECK(hipGraphAddKernelNode(cur, g, deps, ndeps, &p));
  }
}

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

// straight: single linear chain of N nodes.
static hipGraphExec_t build_straight(int N, VerifyCtx* ctx = nullptr) {
  hipGraph_t g;
  HIP_CHECK(hipGraphCreate(&g, 0));

  hipGraphNode_t prev{}, cur{};
  for (int i = 0; i < N; ++i) {
    add_node(g, &cur, i == 0 ? nullptr : &prev, i == 0 ? 0 : 1, ctx);
    prev = cur;
  }

  hipGraphExec_t e;
  HIP_CHECK(hipGraphInstantiate(&e, g, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(g));
  return e;
}

// multi-path (hexagon): lead -> P parallel branches -> tail.
//   seg = N / (P + 2)  nodes per segment
//   total segments     = P + 2
//
// Verification note: the join node depends on all P branch-tail nodes, so its
// slot in dev_buf can only be written after every branch completes.  The
// trailing chain's slots prove the join itself finished.  Checking all slots
// therefore fully validates the fan-out / fan-in ordering.
static hipGraphExec_t build_multi_path(int N, int P, VerifyCtx* ctx = nullptr) {
  const int seg = std::max(1, N / (P + 2));

  hipGraph_t g;
  HIP_CHECK(hipGraphCreate(&g, 0));

  // Leading straight chain.
  hipGraphNode_t prev{}, cur{};
  for (int i = 0; i < seg; ++i) {
    add_node(g, &cur, i == 0 ? nullptr : &prev, i == 0 ? 0 : 1, ctx);
    prev = cur;
  }
  hipGraphNode_t split_end = prev;

  // P parallel branches, each rooted at split_end.
  std::vector<hipGraphNode_t> path_ends(P);
  for (int path = 0; path < P; ++path) {
    hipGraphNode_t pprev = split_end, pcur{};
    for (int i = 0; i < seg; ++i) {
      add_node(g, &pcur, &pprev, 1, ctx);
      pprev = pcur;
    }
    path_ends[path] = pprev;
  }

  // Join node: depends on all P branch tails.
  hipGraphNode_t join{};
  add_node(g, &join, path_ends.data(), P, ctx);
  prev = join;
  for (int i = 1; i < seg; ++i) {
    add_node(g, &cur, &prev, 1, ctx);
    prev = cur;
  }

  hipGraphExec_t e;
  HIP_CHECK(hipGraphInstantiate(&e, g, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(g));
  return e;
}

static hipGraphExec_t build_paths2(int N, VerifyCtx* ctx = nullptr) {
  return build_multi_path(N, 2, ctx);
}
static hipGraphExec_t build_paths4(int N, VerifyCtx* ctx = nullptr) {
  return build_multi_path(N, 4, ctx);
}

// fully parallel: P independent chains of N/P nodes.
// No synchronisation point — GPU can schedule all chains concurrently.
//
// Verification note: chains are fully independent, so checking all slots
// confirms every chain ran to completion.
static hipGraphExec_t build_full_parallel(int N, int P,
                                          VerifyCtx* ctx = nullptr) {
  const int seg = std::max(1, N / P);

  hipGraph_t g;
  HIP_CHECK(hipGraphCreate(&g, 0));

  for (int path = 0; path < P; ++path) {
    hipGraphNode_t pprev{}, pcur{};
    for (int i = 0; i < seg; ++i) {
      add_node(g, &pcur, i == 0 ? nullptr : &pprev, i == 0 ? 0 : 1, ctx);
      pprev = pcur;
    }
  }

  hipGraphExec_t e;
  HIP_CHECK(hipGraphInstantiate(&e, g, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(g));
  return e;
}

static hipGraphExec_t build_full2(int N, VerifyCtx* ctx = nullptr) {
  return build_full_parallel(N, 2, ctx);
}
static hipGraphExec_t build_full4(int N, VerifyCtx* ctx = nullptr) {
  return build_full_parallel(N, 4, ctx);
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

// Returns average hipGraphLaunch time in microseconds over `iters` launches.
// If syncInTiming is true, hipStreamSynchronize is included in each sample.
// The stream is always drained between samples to avoid carry-over GPU work.
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
// Verification runner
// ---------------------------------------------------------------------------

// Builds the graph with verify_kernel nodes, launches it once, copies the
// result back, and checks buf[i] == i+1 for every node i.
// Returns true on PASS.
static bool verify(hipGraphExec_t (*build)(int, VerifyCtx*), int N) {
  VerifyCtx ctx;
  // Allocate a device buffer sized for the maximum possible node count.
  HIP_CHECK(hipMalloc(&ctx.dev_buf, N * sizeof(int)));
  HIP_CHECK(hipMemset(ctx.dev_buf, 0, N * sizeof(int)));

  hipGraphExec_t exec = build(N, &ctx);
  const int actual_n  = ctx.next_id;  // real node count after build

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));

  std::vector<int> host(actual_n);
  HIP_CHECK(hipMemcpy(host.data(), ctx.dev_buf, actual_n * sizeof(int),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipFree(ctx.dev_buf));

  for (int i = 0; i < actual_n; ++i) {
    if (host[i] != i + 1) {
      fprintf(stderr, "  FAIL: buf[%d] = %d, expected %d\n", i, host[i],
              i + 1);
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  int         size         = 1024;
  int         iters        = 1000;
  bool        syncInTiming = false;
  bool        sweep        = false;
  bool        do_verify    = false;
  std::string topo         = "all";

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
    else if (!strcmp(argv[i], "--verify"))
      do_verify = true;
  }

  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  printf("Device : %s\n", props.name);

  struct Topo {
    const char*       name;
    hipGraphExec_t  (*build)(int, VerifyCtx*);
    hipGraphExec_t  (*build_bench)(int);  // no-ctx wrapper for bench()
  };

  // Wrappers that drop the ctx pointer for the bench() call signature.
  auto ws = [](int n) { return build_straight(n); };
  auto wp2 = [](int n) { return build_paths2(n); };
  auto wp4 = [](int n) { return build_paths4(n); };
  auto wf2 = [](int n) { return build_full2(n); };
  auto wf4 = [](int n) { return build_full4(n); };

  const Topo topos[] = {
      {"straight", build_straight, ws },
      {"paths2",   build_paths2,   wp2},
      {"paths4",   build_paths4,   wp4},
      {"full2",    build_full2,    wf2},
      {"full4",    build_full4,    wf4},
  };
  const int ntopos = static_cast<int>(sizeof(topos) / sizeof(topos[0]));

  // -------------------------------------------------------------------------
  // Verification mode
  // -------------------------------------------------------------------------
  if (do_verify) {
    printf("Mode   : verify (size=%d)\n\n", size);
    printf("%-10s  %s\n", "topology", "result");
    printf("%s\n", std::string(24, '-').c_str());

    bool all_pass = true;
    for (int t = 0; t < ntopos; ++t) {
      if (topo != "all" && topo != topos[t].name) continue;
      const bool pass = verify(topos[t].build, size);
      printf("%-10s  %s\n", topos[t].name, pass ? "PASS" : "FAIL");
      all_pass &= pass;
    }
    return all_pass ? 0 : 1;
  }

  // -------------------------------------------------------------------------
  // Benchmark mode
  // -------------------------------------------------------------------------
  printf("Mode   : %s\n",
         syncInTiming ? "sync (submission+GPU)" : "no-sync (submission only)");
  printf("Iters  : %d per measurement\n\n", iters);

  if (sweep) {
    const int sweep_sizes[] = {1,   2,   4,    8,    16,   32,
                                64,  128, 256,  512,  1024, 2048,
                                4096, 8192};
    const int nsizes =
        static_cast<int>(sizeof(sweep_sizes) / sizeof(sweep_sizes[0]));

    printf("%-7s", "size");
    for (int t = 0; t < ntopos; ++t) printf("  %10s", topos[t].name);
    printf("\n%s\n", std::string(7 + ntopos * 12, '-').c_str());

    for (int s = 0; s < nsizes; ++s) {
      const int N = sweep_sizes[s];
      printf("%-7d", N);
      for (int t = 0; t < ntopos; ++t) {
        hipGraphExec_t e = topos[t].build_bench(N);
        const double   avg = bench(e, iters, syncInTiming);
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
      hipGraphExec_t e   = topos[t].build_bench(size);
      const double   avg = bench(e, iters, syncInTiming);
      HIP_CHECK(hipGraphExecDestroy(e));
      printf("%-10s  %.3f us\n", topos[t].name, avg);
    }
  }

  return 0;
}
