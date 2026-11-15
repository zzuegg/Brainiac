/**
 * A GPU backend: runs tensor operations through OpenCL, called directly with the Foreign
 * Function and Memory API. Needs only the OpenCL driver of the graphics card. Run with
 * {@code --enable-native-access=io.github.zzuegg.brainiac.backend.opencl}.
 */
module io.github.zzuegg.brainiac.backend.opencl {
    requires transitive io.github.zzuegg.brainiac.tensor;

    exports io.github.zzuegg.brainiac.backend.opencl;

    provides io.github.zzuegg.brainiac.tensor.spi.Backend
            with io.github.zzuegg.brainiac.backend.opencl.OpenClBackend;
}
