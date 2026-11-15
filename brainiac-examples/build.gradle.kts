plugins {
    id("brainiac.java-application")
}

dependencies {
    implementation(project(":brainiac"))
}

application {
    mainModule = "io.github.zzuegg.brainiac.examples"
    // enables the CPU backend's SIMD kernels (Vector API)
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
    mainClass = "io.github.zzuegg.brainiac.examples.Examples"
}
