plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-nn"))
    api(project(":brainiac-data"))
    testImplementation(project(":brainiac-io"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}
