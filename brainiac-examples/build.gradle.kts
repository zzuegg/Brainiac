plugins {
    id("brainiac.java-application")
}

dependencies {
    implementation(project(":brainiac"))
}

application {
    mainModule = "io.github.zzuegg.brainiac.examples"
    mainClass = "io.github.zzuegg.brainiac.examples.Examples"
}
