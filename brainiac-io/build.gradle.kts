plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-tensor"))
    api(project(":brainiac-data"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}
