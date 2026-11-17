plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-nn"))
    api(project(":brainiac-data"))
    api(project(":brainiac-io"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
    testRuntimeOnly(project(":brainiac-backend-vulkan"))
    testRuntimeOnly(project(":brainiac-backend-opencl"))
}

// -PtestDevice=vulkan (or opencl, or part of a device name) runs the tests that use TestDevice there
tasks.test {
    val device = providers.gradleProperty("testDevice").getOrElse("cpu")
    systemProperty("brainiac.test.device", device)
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (device != "cpu") {
        jvmArgs("-Dbrainiac.vulkan.devices=all", "-Dbrainiac.opencl.devices=all")   // lavapipe and pocl count too
    }
}
