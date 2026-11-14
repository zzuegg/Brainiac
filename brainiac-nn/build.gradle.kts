plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-tensor"))
    api(project(":brainiac-autograd"))
}

dependencies {
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}
