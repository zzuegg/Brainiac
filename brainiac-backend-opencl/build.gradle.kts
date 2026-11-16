plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-tensor"))
    testImplementation(project(":brainiac-nn"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}

// The tests compare every operation with the CPU backend. They are skipped where no
// OpenCL driver is installed; CI installs PoCL (an OpenCL driver for the CPU), which the
// backend only uses when asked to consider CPU devices.
tasks.test {
    jvmArgs(
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules", "jdk.incubator.vector",
        "-Dbrainiac.opencl.devices=all",
        // big tiles for every product of at least 64 x 64, so small test shapes cover them
        "-Dbrainiac.opencl.bigTileGroups=1",
    )
}

// Calling native code through the FFM API uses restricted methods by design.
tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.add("-Xlint:-restricted")
}
