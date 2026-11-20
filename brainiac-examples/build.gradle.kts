plugins {
    id("brainiac.java-application")
}

dependencies {
    implementation(project(":brainiac"))
    implementation(project(":brainiac-backend-vulkan"))
    implementation(project(":brainiac-evolve"))
    // found through ServiceLoader, so that --device cuda works where there is an NVIDIA GPU and --device hip on an AMD one
    runtimeOnly(project(":brainiac-backend-cuda"))
    runtimeOnly(project(":brainiac-backend-hip"))
}

application {
    mainModule = "io.github.zzuegg.brainiac.examples"
    // enables the CPU backend's SIMD kernels (Vector API) and lets LWJGL (for the
    // Vulkan backend) and the CUDA and HIP backends call the GPU driver
    applicationDefaultJvmArgs = listOf(
        "--add-modules", "jdk.incubator.vector,io.github.zzuegg.brainiac.backend.cuda,io.github.zzuegg.brainiac.backend.hip",
        "--enable-native-access=org.lwjgl,io.github.zzuegg.brainiac.backend.cuda,io.github.zzuegg.brainiac.backend.hip",
    )
    mainClass = "io.github.zzuegg.brainiac.examples.Examples"
}

tasks.test {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
