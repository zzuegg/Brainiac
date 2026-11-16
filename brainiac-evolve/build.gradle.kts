plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-train"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}

dependencies {
    // the GPU tests run on PoCL in CI and are skipped where no OpenCL driver is installed
    testRuntimeOnly(project(":brainiac-backend-opencl"))
}

tasks.test {
    jvmArgs("--enable-native-access=ALL-UNNAMED", "-Dbrainiac.opencl.devices=all")
}
