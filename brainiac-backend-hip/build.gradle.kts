plugins {
    id("brainiac.java-library")
}

// No dependencies beyond the tensor API: the HIP runtime, hipRTC and rocBLAS are called through
// the Foreign Function and Memory API and found on the machine at run time.
dependencies {
    api(project(":brainiac-tensor"))
    testImplementation(project(":brainiac-nn"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}

// The tests compare every operation with the CPU backend. Without an AMD GPU and the HIP
// runtime (as on CI) the device tests are skipped and only the host-side ones run.
tasks.test {
    // the benchmark compares 8192 x 8192 products with the CPU's arrays
    maxHeapSize = "8g"
    jvmArgs("--enable-native-access=ALL-UNNAMED", "--add-modules", "jdk.incubator.vector")
    // rocBLAS and hipRTC load their own libraries from the HIP SDK's bin folder, which Windows
    // only searches when it is on the PATH
    val hipPath = System.getenv("HIP_PATH")
    if (System.getProperty("os.name").startsWith("Windows") && !hipPath.isNullOrBlank()) {
        val bin = File(hipPath, "bin")
        if (bin.isDirectory) {
            val key = System.getenv().keys.firstOrNull { it.equals("PATH", ignoreCase = true) } ?: "PATH"
            environment(key, bin.absolutePath + File.pathSeparator + (System.getenv(key) ?: ""))
        }
    }
    // -Pbrainiac.hip.debug=true waits after every operation, so errors name their operation
    if (providers.gradleProperty("brainiac.hip.debug").orNull == "true") {
        jvmArgs("-Dbrainiac.hip.debug=true")
    }
    // -Pbrainiac.hip.trace=true prints every HIP call before it is made, to find a hang
    if (providers.gradleProperty("brainiac.hip.trace").orNull == "true") {
        jvmArgs("-Dbrainiac.hip.trace=true")
    }
    // -Pbrainiac.hip.critical=true links the per-operation calls as critical downcalls
    if (providers.gradleProperty("brainiac.hip.critical").orNull == "true") {
        jvmArgs("-Dbrainiac.hip.critical=true")
    }
    // -Pbrainiac.hip.rocblas=none multiplies with the backend's own kernel
    providers.gradleProperty("brainiac.hip.rocblas").orNull?.let { jvmArgs("-Dbrainiac.hip.rocblas=$it") }
    // -Pbench prints matrix product speeds (HipBenchmarkTest)
    if (providers.gradleProperty("bench").isPresent) {
        jvmArgs("-Dbrainiac.bench=true")
    }
    testLogging.showStandardStreams = true
    // name each test as it starts, so a test that never ends shows which one it is
    testLogging.events("started", "passed", "skipped", "failed")
}
