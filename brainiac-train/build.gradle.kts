plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-nn"))
    api(project(":brainiac-data"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}
