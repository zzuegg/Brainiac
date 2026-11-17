plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-nn"))
    api(project(":brainiac-io"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
    testRuntimeOnly(project(":brainiac-backend-vulkan"))
    testRuntimeOnly(project(":brainiac-backend-opencl"))
}

// -PtestDevice=vulkan (or opencl, or part of a device name) runs the tests on that device
tasks.test {
    val device = providers.gradleProperty("testDevice").getOrElse("cpu")
    systemProperty("brainiac.test.device", device)
    // -PtestModels=<folder of Hugging Face models and .gguf files> runs PretrainedModelsTest
    systemProperty("brainiac.test.models", providers.gradleProperty("testModels").getOrElse(""))
    systemProperty("brainiac.test.dtype", providers.gradleProperty("testDtype").getOrElse("f32"))
    providers.gradleProperty("testModels").orNull?.let {
        maxHeapSize = "8g"
        testLogging.showStandardStreams = true
    }
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (device != "cpu") {
        jvmArgs("-Dbrainiac.vulkan.devices=all", "-Dbrainiac.opencl.devices=all")   // lavapipe and pocl count too
    }
}
