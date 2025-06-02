#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

struct Result {
  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

struct Options {

  bool help;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes;
  int batch_count;
  float alpha;
  float beta;

  bool reference_check;
  int iterations;
  
  Options():
    help(false),
    problem_sizes({}),
    batch_count(1),
    reference_check(true),
    iterations(20),
    alpha(1),
    beta() { }

  bool valid() {
    return true;
  }

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    int mn;
    std::vector<int> k_vals{};
    cmd.get_cmd_line_argument("mn", mn);
    cmd.get_cmd_line_arguments("k", k_vals);
    for (auto k : k_vals) {
        problem_sizes.push_back({mn, mn, k});
    }

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  std::ostream & print_usage(std::ostream &out) const {
    out << "CUTLASS TF32 Tall and Skinny Benchmark\n\n"
      << "  This benchmark uses the CUTLASS Library to test TF32 tensorop GEMM computations on tall and skinny matrices.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --mn=<int>                   GEMM M and N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    return out;
  }

  double gflops(cutlass::gemm::GemmCoord problem_size, double runtime_s) const {
    int64_t fmas = problem_size.product() * batch_count;
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;                        // <- data type of elements in input matrix A
using ElementInputB = float;                        // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm80;

using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

constexpr int NumStages = 4;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

int run(Options &options) {
    std::vector<cutlass::gemm::GemmCoord> problem_sizes = options.problem_sizes;

    for (auto problem_size : problem_sizes) {
        cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
            problem_size.mk());
        cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
            problem_size.kn());
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
            problem_size.mn());
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
            problem_size.mn());
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
            problem_size.mn());

        cutlass::reference::host::TensorFillRandomUniform(
            tensor_a.host_view(),
            1,
            ElementInputA(4),
            ElementInputA(-4),
            0);
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_b.host_view(),
            1,
            ElementInputB(4),
            ElementInputB(-4),
            0);
        cutlass::reference::host::TensorFillRandomUniform(
            tensor_c.host_view(),
            1,
            ElementOutput(4),
            ElementOutput(-4),
            0);
        cutlass::reference::host::TensorFill(
            tensor_d.host_view());
        cutlass::reference::host::TensorFill(
            tensor_ref_d.host_view());

        tensor_a.sync_device();
        tensor_b.sync_device();
        tensor_c.sync_device();
        tensor_d.sync_device();
        tensor_ref_d.sync_device();

        ElementComputeEpilogue alpha = ElementComputeEpilogue(options.alpha);
        ElementComputeEpilogue beta = ElementComputeEpilogue(options.beta);

        int split_k_slices = 1;

        typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                            tensor_a.device_ref(),  // <- reference to matrix A on device
                                            tensor_b.device_ref(),  // <- reference to matrix B on device
                                            tensor_c.device_ref(),  // <- reference to matrix C on device
                                            tensor_d.device_ref(),  // <- reference to matrix D on device
                                            {alpha, beta},          // <- tuple of alpha and beta
                                            split_k_slices};        // <- k-dimension split factor

        // Using the arguments, query for extra workspace required for matrix multiplication computation
        size_t workspace_size = Gemm::get_workspace_size(arguments);

        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // Instantiate CUTLASS kernel depending on templates
        Gemm gemm_op;

        // Check the problem size is supported or not 
        cutlass::Status status = gemm_op.can_implement(arguments);
        CUTLASS_CHECK(status);

        // Initialize CUTLASS kernel with arguments and workspace pointer
        status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);

        // Result structure
        Result result;

        cudaEvent_t events[2];

        for (auto & event : events) {
            result.error = cudaEventCreate(&event);
            if (result.error != cudaSuccess) {
            std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
            return -1;
            }
        }

        result.error = cudaEventRecord(events[0]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
            return -1;
        }

        for (int iter = 0; iter < options.iterations; ++iter) {
            status = gemm_op();
            CUTLASS_CHECK(status);
        }

        result.error = cudaEventRecord(events[1]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
            return -1;
        }

        result.error = cudaEventSynchronize(events[1]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
            return -1;
        }

        float runtime_ms = 0;
        result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
        if (result.error != cudaSuccess) {
            std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
            return -1;
        }

        result.runtime_ms = double(runtime_ms) / double(options.iterations);
        result.gflops = options.gflops(problem_size, result.runtime_ms / 1000.0);

        for (auto event : events) {
            (void)cudaEventDestroy(event);
        }

        // Create instantiation for device reference gemm kernel
        cutlass::reference::device::Gemm<ElementInputA,
                                        LayoutInputA,
                                        ElementInputB,
                                        LayoutInputB,
                                        ElementOutput,
                                        LayoutOutput,
                                        ElementComputeEpilogue,
                                        ElementComputeEpilogue>
            gemm_device;

        // Launch device reference gemm kernel
        gemm_device(problem_size,
                    alpha,
                    tensor_a.device_ref(),
                    tensor_b.device_ref(),
                    beta,
                    tensor_c.device_ref(),
                    tensor_ref_d.device_ref());

        // Wait for kernels to finish
        cudaDeviceSynchronize();

        // Copy output data from CUTLASS and reference kernel to host for comparison
        tensor_d.sync_host();
        tensor_ref_d.sync_host();

        // Check if output from CUTLASS kernel and reference kernel are equal or not
        bool passed = cutlass::reference::host::TensorEquals(
            tensor_d.host_view(),
            tensor_ref_d.host_view());

        if (passed) {
            std::cout << (passed ? "Passed" : "Failed") << ','
                << problem_size.m() << ',' << problem_size.n() << ',' << problem_size.k() << ','
                << result.runtime_ms << ','
                << result.gflops << std::endl;
        } else {
            std::cerr << "Failed with the following problem size: "
                << problem_size.m() << ','
                << problem_size.n() << ','
                << problem_size.k() << std::endl;

            return -1;
        }
    }
    return 0;
}

int main(int argc, const char **argv) {
  
  bool notSupported = false;

  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    return 0;
  }

  Options options;
  options.parse(argc, argv);

  if (options.problem_sizes.size() < 1) {
    std::cerr << "No problem sizes provided" << std::endl;
  }

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  std::cout << "passed,m,n,k,runtime(ms),gflops" << std::endl;
  return run(options);
}