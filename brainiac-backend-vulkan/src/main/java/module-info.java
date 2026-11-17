/**
 * A GPU backend on Vulkan compute shaders, called through LWJGL. Needs a Vulkan 1.2 driver
 * and the LWJGL natives for the platform. Run with
 * {@code --enable-native-access=org.lwjgl}.
 */
module io.github.zzuegg.brainiac.backend.vulkan {
    requires transitive io.github.zzuegg.brainiac.tensor;
    requires org.lwjgl;
    requires org.lwjgl.vulkan;
    // the shared libraries of the platform's natives jar (the module name is the same on every platform)
    requires org.lwjgl.natives;
    // compiles the shaders of fused element-wise chains
    requires org.lwjgl.shaderc;
    requires org.lwjgl.shaderc.natives;

    exports io.github.zzuegg.brainiac.backend.vulkan;

    provides io.github.zzuegg.brainiac.tensor.spi.Backend
            with io.github.zzuegg.brainiac.backend.vulkan.VulkanBackend;
}
