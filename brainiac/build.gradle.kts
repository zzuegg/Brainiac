plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-tensor"))
    api(project(":brainiac-autograd"))
    api(project(":brainiac-nn"))
    api(project(":brainiac-train"))
    api(project(":brainiac-data"))
    api(project(":brainiac-rl"))
    api(project(":brainiac-evolve"))
}

dependencies {
    implementation(project(":brainiac-backend-cpu"))
}
