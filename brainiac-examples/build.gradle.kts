plugins {
    id("brainiac.java-application")
}

dependencies {
    implementation(project(":brainiac"))
    implementation(project(":brainiac-backend-opencl"))
    implementation(project(":brainiac-backend-vulkan"))
    implementation(project(":brainiac-evolve"))
}

application {
    mainModule = "io.github.zzuegg.brainiac.examples"
    // enables the CPU backend's SIMD kernels (Vector API) and lets the OpenCL backend and
    // LWJGL (for the Vulkan backend) call the GPU driver
    applicationDefaultJvmArgs = listOf(
        "--add-modules", "jdk.incubator.vector",
        "--enable-native-access=io.github.zzuegg.brainiac.backend.opencl,org.lwjgl",
    )
    mainClass = "io.github.zzuegg.brainiac.examples.Examples"
}

tasks.test {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
