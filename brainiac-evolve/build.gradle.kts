plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-train"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}

dependencies {
    // the GPU tests run on lavapipe in CI and are skipped where no Vulkan driver is installed
    testRuntimeOnly(project(":brainiac-backend-vulkan"))
}

tasks.test {
    jvmArgs("--enable-native-access=ALL-UNNAMED", "-Dbrainiac.vulkan.devices=all")
}
