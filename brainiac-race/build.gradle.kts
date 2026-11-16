plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-rl"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}
