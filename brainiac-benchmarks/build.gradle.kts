plugins {
    id("brainiac.java-application")
}

dependencies {
    implementation(project(":brainiac"))
    // for --device: the GPU backends, found through the service loader
    runtimeOnly(project(":brainiac-backend-opencl"))
    runtimeOnly(project(":brainiac-backend-vulkan"))
}

application {
    mainModule = "io.github.zzuegg.brainiac.benchmarks"
    // enables the CPU backend's SIMD kernels (Vector API), resolves the GPU backends and
    // lets them call the driver
    applicationDefaultJvmArgs = listOf(
        "--add-modules", "jdk.incubator.vector,io.github.zzuegg.brainiac.backend.opencl,"
                + "io.github.zzuegg.brainiac.backend.vulkan",
        "--enable-native-access=io.github.zzuegg.brainiac.backend.opencl,org.lwjgl",
    )
    mainClass = "io.github.zzuegg.brainiac.benchmarks.Benchmarks"
}
