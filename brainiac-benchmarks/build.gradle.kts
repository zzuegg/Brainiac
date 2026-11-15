plugins {
    id("brainiac.java-application")
}

dependencies {
    implementation(project(":brainiac"))
}

application {
    mainModule = "io.github.zzuegg.brainiac.benchmarks"
    mainClass = "io.github.zzuegg.brainiac.benchmarks.Benchmarks"
}
