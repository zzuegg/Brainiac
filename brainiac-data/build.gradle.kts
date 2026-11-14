plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-tensor"))
}

dependencies {
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}
