plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-nn"))
    api(project(":brainiac-io"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}
