plugins {
    id("brainiac.java-library")
}

// No dependencies beyond the tensor API: the CUDA driver, NVRTC and cuBLAS are called through
// the Foreign Function and Memory API and found on the machine at run time.
dependencies {
    api(project(":brainiac-tensor"))
    testImplementation(project(":brainiac-nn"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}

// The tests compare every operation with the CPU backend. Without an NVIDIA driver (as on
// CI) the device tests are skipped and only the host-side ones run.
tasks.test {
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector")
    // -Pbrainiac.cuda.debug=true waits after every operation, so errors name their operation
    if (providers.gradleProperty("brainiac.cuda.debug").orNull == "true") {
        jvmArgs("-Dbrainiac.cuda.debug=true")
    }
    // -Pbrainiac.cuda.critical=true links the per-operation calls as critical downcalls
    if (providers.gradleProperty("brainiac.cuda.critical").orNull == "true") {
        jvmArgs("-Dbrainiac.cuda.critical=true")
    }
    // -Pbench prints matrix product speeds (CudaBenchmarkTest)
    if (providers.gradleProperty("bench").isPresent) {
        jvmArgs("-Dbrainiac.bench=true")
    }
    testLogging.showStandardStreams = true
}
