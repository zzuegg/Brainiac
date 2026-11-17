plugins {
    id("brainiac.java-library")
}

// Experimental, untested research ideas built on the core library. Nothing in the core
// depends on this module, and its API may change or disappear at any time.
dependencies {
    api(project(":brainiac-nn"))
    api(project(":brainiac-train"))
    api(project(":brainiac-data"))
    api(project(":brainiac-rl"))
    api(project(":brainiac-evolve"))
    api(project(":brainiac-llm"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}

tasks.test {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
