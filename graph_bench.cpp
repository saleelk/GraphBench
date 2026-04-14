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
// Replaces null_kernel with a reduction-based ordering check:
//
//   verify_kernel writes:
//     buf[nodeId] = sum(buf[dep_ids]) + 1
//
// For a root node (no deps) this is 1.  For a chain node it is predecessor+1.
// For the hexagon join it is the sum of all branch-tail values plus 1.
//
// Expected values are computed on the CPU at build time using the same
// recurrence.  Only the graph's exit node(s) are checked:
//
//   straight / paths*  : one exit (tail of trailing chain)
//   full2 / full4      : one exit per independent chain
//
// If any node ran before its dependencies, its buf slot holds a smaller value
// than expected (it read zeros instead of its predecessor's value), and that
// propagates to the exit — so a single wrong exit value catches the race.
//
// Example  paths2, N=12, seg=3:
//   Lead:    buf[0]=1  buf[1]=2  buf[2]=3
//   Branch0: buf[3]=4  buf[4]=5  buf[5]=6
//   Branch1: buf[6]=4  buf[7]=5  buf[8]=6
//   Join:    buf[9] = buf[5]+buf[8]+1 = 13  (wrong if either branch not done)
//   Tail:    buf[10]=14  buf[11]=15
//   Check:   buf[11] == 15
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
//   --verify          Run ordering correctness check instead of timing

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
#include <unordered_map>
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

// Reduction-based ordering kernel.
//
// Writes buf[nodeId] = buf[d0] + buf[d1] + buf[d2] + buf[d3] + 1
// where d0..d3 are the IDs of this node's graph predecessors (unused slots
// are 0 and excluded via ndeps).
//
// If any predecessor has not yet written its slot (i.e. ran out of order),
// buf[dep] is still 0 and the computed value will be smaller than expected.
// That deficit propagates through the chain to the exit node, where a single
// comparison catches the ordering violation.
//
// Supports up to 4 predecessor IDs, which is enough for paths4 (join has 4).
__global__ void verify_kernel(int* buf, int nodeId,
                               int d0, int d1, int d2, int d3, int ndeps) {
  int val = 1;
  if (ndeps > 0) val += buf[d0];
  if (ndeps > 1) val += buf[d1];
  if (ndeps > 2) val += buf[d2];
  if (ndeps > 3) val += buf[d3];
  buf[nodeId] = val;
}

// ---------------------------------------------------------------------------
// Verification context
// ---------------------------------------------------------------------------

// Passed to graph builders when --verify is active.
//
// add_node() auto-assigns a node ID, registers the handle->ID mapping (so
// that later nodes can resolve their predecessor IDs by handle), computes the
// expected output value using the same recurrence as verify_kernel, and wires
// up verify_kernel in the graph node.
//
// After the build, exits[] contains the exit node ID(s) and expected[] holds
// the correct value for every node.  Only the exit values are checked.
struct VerifyCtx {
  int*  dev_buf  = nullptr;  // device buffer, size >= next_id (caller-owned)
  int   next_id  = 0;        // auto-incremented as nodes are added

  std::vector<int>                             expected;    // expected[nodeId]
  std::vector<int>                             exits;       // exit node IDs
  std::unordered_map<hipGraphNode_t, int>      node_to_id;  // handle -> ID
};

// ---------------------------------------------------------------------------
// Node helper
// ---------------------------------------------------------------------------

// Add one kernel node to graph g depending on deps[0..ndeps-1].
//
// Non-verify mode (ctx == nullptr): wires null_kernel, returns -1.
//
// Verify mode: assigns the next ID, resolves predecessor handles to IDs via
// ctx->node_to_id, computes the expected value, sets up verify_kernel with
// the dep IDs as arguments, registers the new handle, and returns the ID.
// The caller is responsible for adding the returned ID to ctx->exits if this
// is an exit node.
static int add_node(hipGraph_t g, hipGraphNode_t* cur,
                    const hipGraphNode_t* deps, int ndeps,
                    VerifyCtx* ctx) {
  hipKernelNodeParams p{};
  p.gridDim  = {1, 1, 1};
  p.blockDim = {1, 1, 1};

  if (!ctx) {
    p.func = reinterpret_cast<void*>(null_kernel);
    HIP_CHECK(hipGraphAddKernelNode(cur, g, deps, ndeps, &p));
    return -1;
  }

  // Resolve predecessor handles to IDs and compute expected value on CPU.
  int d[4]  = {0, 0, 0, 0};
  int exp   = 1;
  for (int i = 0; i < ndeps && i < 4; ++i) {
    d[i] = ctx->node_to_id.at(deps[i]);
    exp += ctx->expected[d[i]];
  }

  int id = ctx->next_id++;
  ctx->expected.push_back(exp);

  // hipGraphAddKernelNode copies args immediately, so locals are safe here.
  void* args[] = {reinterpret_cast<void*>(&ctx->dev_buf),
                  reinterpret_cast<void*>(&id),
                  reinterpret_cast<void*>(&d[0]),
                  reinterpret_cast<void*>(&d[1]),
                  reinterpret_cast<void*>(&d[2]),
                  reinterpret_cast<void*>(&d[3]),
                  reinterpret_cast<void*>(&ndeps)};
  p.func         = reinterpret_cast<void*>(verify_kernel);
  p.kernelParams = args;
  HIP_CHECK(hipGraphAddKernelNode(cur, g, deps, ndeps, &p));

  ctx->node_to_id[*cur] = id;  // register after hipGraphAddKernelNode sets *cur
  return id;
}

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

// straight: single linear chain of N nodes.
// Exit: the last node.
static hipGraphExec_t build_straight(int N, VerifyCtx* ctx = nullptr) {
  hipGraph_t g;
  HIP_CHECK(hipGraphCreate(&g, 0));

  hipGraphNode_t prev{}, cur{};
  int prev_id = -1, cur_id = -1;
  for (int i = 0; i < N; ++i) {
    cur_id  = add_node(g, &cur, i == 0 ? nullptr : &prev, i == 0 ? 0 : 1, ctx);
    prev    = cur;
    prev_id = cur_id;
  }
  if (ctx && cur_id >= 0) ctx->exits.push_back(cur_id);

  hipGraphExec_t e;
  HIP_CHECK(hipGraphInstantiate(&e, g, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(g));
  return e;
}

// multi-path (hexagon): lead -> P parallel branches -> tail.
//   seg = N / (P + 2)  nodes per segment
//   total segments     = P + 2
//
// Ordering guarantee encoded in expected values:
//   join's expected = sum(branch_tail_expected[0..P-1]) + 1
// If any branch tail ran after the join, its buf slot is 0 at join time,
// making the join's actual value smaller than expected. That deficit
// propagates through the tail chain to the single exit node.
static hipGraphExec_t build_multi_path(int N, int P, VerifyCtx* ctx = nullptr) {
  const int seg = std::max(1, N / (P + 2));

  hipGraph_t g;
  HIP_CHECK(hipGraphCreate(&g, 0));

  // Leading straight chain.
  hipGraphNode_t prev{}, cur{};
  int prev_id = -1, cur_id = -1;
  for (int i = 0; i < seg; ++i) {
    cur_id  = add_node(g, &cur, i == 0 ? nullptr : &prev, i == 0 ? 0 : 1, ctx);
    prev    = cur;
    prev_id = cur_id;
  }
  hipGraphNode_t split_node    = prev;
  int            split_node_id = cur_id;

  // P parallel branches, each rooted at split_node.
  std::vector<hipGraphNode_t> path_ends(P);
  std::vector<int>            path_end_ids(P);
  for (int path = 0; path < P; ++path) {
    hipGraphNode_t pprev    = split_node;
    int            pprev_id = split_node_id;
    hipGraphNode_t pcur{};
    int            pcur_id = -1;
    for (int i = 0; i < seg; ++i) {
      pcur_id = add_node(g, &pcur, &pprev, 1, ctx);
      pprev   = pcur;
      pprev_id = pcur_id;
    }
    path_ends[path]     = pprev;
    path_end_ids[path]  = pprev_id;
  }

  // Join node: depends on all P branch tails.
  hipGraphNode_t join{};
  int join_id = add_node(g, &join, path_ends.data(), P, ctx);
  prev    = join;
  prev_id = join_id;

  // Trailing straight chain.
  for (int i = 1; i < seg; ++i) {
    cur_id  = add_node(g, &cur, &prev, 1, ctx);
    prev    = cur;
    prev_id = cur_id;
  }
  if (ctx && prev_id >= 0) ctx->exits.push_back(prev_id);

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
// Exit: last node of each chain (P exits total).
static hipGraphExec_t build_full_parallel(int N, int P,
                                          VerifyCtx* ctx = nullptr) {
  const int seg = std::max(1, N / P);

  hipGraph_t g;
  HIP_CHECK(hipGraphCreate(&g, 0));

  for (int path = 0; path < P; ++path) {
    hipGraphNode_t pprev{}, pcur{};
    int            pprev_id = -1, pcur_id = -1;
    for (int i = 0; i < seg; ++i) {
      pcur_id  = add_node(g, &pcur,
                          i == 0 ? nullptr : &pprev, i == 0 ? 0 : 1, ctx);
      pprev    = pcur;
      pprev_id = pcur_id;
    }
    if (ctx && pprev_id >= 0) ctx->exits.push_back(pprev_id);
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

static double bench(hipGraphExec_t exec, int iters, bool syncInTiming) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

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

// Builds the graph with verify_kernel nodes, launches once, copies back the
// device buffer, and checks only the exit node(s) against their expected
// values.  A wrong exit value means some node ran before a dependency.
static bool verify(hipGraphExec_t (*build)(int, VerifyCtx*), int N,
                   const char* name) {
  VerifyCtx ctx;
  // Over-allocate: actual node count (after integer-division seg rounding)
  // may be slightly less than N, but never more.
  HIP_CHECK(hipMalloc(&ctx.dev_buf, N * sizeof(int)));
  HIP_CHECK(hipMemset(ctx.dev_buf, 0, N * sizeof(int)));

  hipGraphExec_t exec = build(N, &ctx);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));

  // Copy only the slots we need.
  const int total = ctx.next_id;
  std::vector<int> host(total);
  HIP_CHECK(hipMemcpy(host.data(), ctx.dev_buf, total * sizeof(int),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipFree(ctx.dev_buf));

  bool pass = true;
  for (int exit_id : ctx.exits) {
    const int got      = host[exit_id];
    const int expected = ctx.expected[exit_id];
    if (got != expected) {
      fprintf(stderr,
              "  [%s] FAIL exit node %d: got %d, expected %d "
              "(ordering violation detected)\n",
              name, exit_id, got, expected);
      pass = false;
    }
  }
  return pass;
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
    hipGraphExec_t  (*build)(int, VerifyCtx*);  // verify-capable builder
    hipGraphExec_t  (*build_bench)(int);         // no-ctx wrapper for bench()
  };

  auto ws  = [](int n) { return build_straight(n); };
  auto wp2 = [](int n) { return build_paths2(n);   };
  auto wp4 = [](int n) { return build_paths4(n);   };
  auto wf2 = [](int n) { return build_full2(n);    };
  auto wf4 = [](int n) { return build_full4(n);    };

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
    printf("Mode   : verify (reduction-based ordering check, size=%d)\n\n",
           size);
    printf("%-10s  %-6s  %s\n", "topology", "exits", "result");
    printf("%s\n", std::string(32, '-').c_str());

    bool all_pass = true;
    for (int t = 0; t < ntopos; ++t) {
      if (topo != "all" && topo != topos[t].name) continue;

      // Dry-run build (no device alloc) just to count exits.
      VerifyCtx dummy;
      dummy.dev_buf = nullptr;
      hipGraphExec_t tmp = topos[t].build(size, &dummy);
      HIP_CHECK(hipGraphExecDestroy(tmp));
      const int nexits = static_cast<int>(dummy.exits.size());

      const bool pass = verify(topos[t].build, size, topos[t].name);
      printf("%-10s  %-6d  %s\n", topos[t].name, nexits,
             pass ? "PASS" : "FAIL");
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
        hipGraphExec_t e   = topos[t].build_bench(N);
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
