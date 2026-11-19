plugins {
    id("brainiac.java-application")
}

// Brainiac Studio: a local web app for using, building and training models. The server is
// the JDK's own HTTP server, so the studio adds no dependencies; the web page is plain
// HTML, CSS and JavaScript served from this module's resources.
dependencies {
    implementation(project(":brainiac"))
    implementation(project(":brainiac-cloud"))
    runtimeOnly(project(":brainiac-backend-vulkan"))
    // the other GPU backends, chosen in the studio's settings: CUDA on NVIDIA cards, HIP on AMD ones
    runtimeOnly(project(":brainiac-backend-cuda"))
    runtimeOnly(project(":brainiac-backend-hip"))
    runtimeOnly(project(":brainiac-experiments"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}

application {
    mainModule = "io.github.zzuegg.brainiac.studio"
    mainClass = "io.github.zzuegg.brainiac.studio.Studio"
    // ALL-MODULE-PATH resolves modules nothing requires, so capabilities from optional
    // modules (GPU backends, experiments) are found at startup
    applicationDefaultJvmArgs = listOf(
        "--add-modules", "jdk.incubator.vector,ALL-MODULE-PATH",
        "--enable-native-access=org.lwjgl,io.github.zzuegg.brainiac.backend.cuda,io.github.zzuegg.brainiac.backend.hip",
    )
}

tasks.named<JavaExec>("run") {
    standardInput = System.`in`
    // ./models means the models folder at the top of the repository
    workingDir = rootProject.projectDir
}

tasks.test {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
