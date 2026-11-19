/**
 * A GPU backend for AMD GPUs on HIP (ROCm), called through the Foreign Function and Memory API
 * (no JNI, no third-party libraries): the HIP runtime, hipRTC to compile the kernels at run
 * time and, when installed, rocBLAS for matrix products. Run with
 * {@code --enable-native-access=io.github.zzuegg.brainiac.backend.hip}.
 */
module io.github.zzuegg.brainiac.backend.hip {
    requires transitive io.github.zzuegg.brainiac.tensor;

    exports io.github.zzuegg.brainiac.backend.hip;

    provides io.github.zzuegg.brainiac.tensor.spi.Backend
            with io.github.zzuegg.brainiac.backend.hip.HipBackend;
}
