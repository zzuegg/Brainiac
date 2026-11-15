plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-tensor"))
}

// The optional SIMD kernels use the incubating Vector API: compiling against it is
// expected, so only that warning is silenced.
tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.add("-Xlint:-incubating")
}

// Test both the SIMD kernels and the plain fallback.
tasks.test {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}
