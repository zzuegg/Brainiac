plugins {
    id("brainiac.java-application")
}

dependencies {
    implementation(project(":brainiac"))
    implementation(project(":brainiac-backend-vulkan"))
    implementation(project(":brainiac-evolve"))
    // found through ServiceLoader, so that --device cuda works where there is an NVIDIA GPU
    runtimeOnly(project(":brainiac-backend-cuda"))
}

application {
    mainModule = "io.github.zzuegg.brainiac.examples"
    // enables the CPU backend's SIMD kernels (Vector API) and lets LWJGL (for the
    // Vulkan backend) and the CUDA backend call the GPU driver
    applicationDefaultJvmArgs = listOf(
        "--add-modules", "jdk.incubator.vector,io.github.zzuegg.brainiac.backend.cuda",
        "--enable-native-access=org.lwjgl,io.github.zzuegg.brainiac.backend.cuda",
    )
    mainClass = "io.github.zzuegg.brainiac.examples.Examples"
}

tasks.test {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
