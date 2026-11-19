plugins {
    id("brainiac.java-application")
}

dependencies {
    implementation(project(":brainiac"))
    // for --device: the GPU backends, found through the service loader
    runtimeOnly(project(":brainiac-backend-vulkan"))
    runtimeOnly(project(":brainiac-backend-cuda"))
    runtimeOnly(project(":brainiac-backend-hip"))
}

application {
    mainModule = "io.github.zzuegg.brainiac.benchmarks"
    // enables the CPU backend's SIMD kernels (Vector API), resolves the GPU backends and
    // lets them call the driver
    applicationDefaultJvmArgs = listOf(
        "--add-modules", "jdk.incubator.vector,io.github.zzuegg.brainiac.backend.vulkan,io.github.zzuegg.brainiac.backend.cuda,io.github.zzuegg.brainiac.backend.hip",
        "--enable-native-access=org.lwjgl,io.github.zzuegg.brainiac.backend.cuda,io.github.zzuegg.brainiac.backend.hip",
    )
    mainClass = "io.github.zzuegg.brainiac.benchmarks.Benchmarks"
}
