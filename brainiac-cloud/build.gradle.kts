plugins {
    id("brainiac.java-library")
    application
}

// Renting GPUs on Vast.ai: a small client for its REST API (the JDK's HTTP client and the
// JSON reader from brainiac-io, no other dependencies) and a command line that rents one
// instance, ships this checkout to it, runs cloud/vast/setup.sh there and destroys it again.
dependencies {
    api(project(":brainiac-io"))
}

application {
    mainModule = "io.github.zzuegg.brainiac.cloud"
    mainClass = "io.github.zzuegg.brainiac.cloud.VastCli"
}

tasks.named<JavaExec>("run") {
    // the launcher asks before it rents anything
    standardInput = System.`in`
    // git archive and cloud/vast/setup.sh are found from the top of the repository
    workingDir = rootProject.projectDir
}
