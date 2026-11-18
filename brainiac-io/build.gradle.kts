plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-tensor"))
    api(project(":brainiac-data"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
    testRuntimeOnly(project(":brainiac-backend-vulkan"))
    testRuntimeOnly(project(":brainiac-backend-cuda"))
}

// -PtestDevice=vulkan or cuda (or part of a device name) runs the tests on that device
tasks.test {
    val device = providers.gradleProperty("testDevice").getOrElse("cpu")
    systemProperty("brainiac.test.device", device)
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (device != "cpu") {
        jvmArgs("-Dbrainiac.vulkan.devices=all")   // lavapipe counts too
    }
}
