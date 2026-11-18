plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-nn"))
    api(project(":brainiac-io"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
    testRuntimeOnly(project(":brainiac-backend-vulkan"))
}

// -PtestDevice=vulkan (or part of a device name) runs the tests on that device
// -PtestModels=<folder with whisper-* and mms-tts-* model folders> runs PretrainedAudioTest
tasks.test {
    val device = providers.gradleProperty("testDevice").getOrElse("cpu")
    systemProperty("brainiac.test.device", device)
    systemProperty("brainiac.test.models", providers.gradleProperty("testModels").getOrElse(""))
    providers.gradleProperty("testModels").orNull?.let {
        maxHeapSize = "4g"
        testLogging.showStandardStreams = true
    }
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (device != "cpu") {
        jvmArgs("-Dbrainiac.vulkan.devices=all")   // lavapipe counts too
    }
}
