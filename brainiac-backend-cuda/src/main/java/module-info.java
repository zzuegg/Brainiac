/**
 * A GPU backend for NVIDIA GPUs on CUDA, called through the Foreign Function and Memory API
 * (no JNI, no third-party libraries): the CUDA driver API, NVRTC to compile the kernels at run
 * time and, when installed, cuBLAS for matrix products. Run with
 * {@code --enable-native-access=io.github.zzuegg.brainiac.backend.cuda}.
 */
module io.github.zzuegg.brainiac.backend.cuda {
    requires transitive io.github.zzuegg.brainiac.tensor;

    exports io.github.zzuegg.brainiac.backend.cuda;

    provides io.github.zzuegg.brainiac.tensor.spi.Backend
            with io.github.zzuegg.brainiac.backend.cuda.CudaBackend;
}
