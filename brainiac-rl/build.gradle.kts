plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-train"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}
