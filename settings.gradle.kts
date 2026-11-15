pluginManagement {
    includeBuild("build-logic")
}

dependencyResolutionManagement {
    repositoriesMode = RepositoriesMode.FAIL_ON_PROJECT_REPOS
    repositories {
        mavenCentral()
    }
}

rootProject.name = "brainiac-root"

include(
    "brainiac-tensor",
    "brainiac-backend-cpu",
    "brainiac-autograd",
    "brainiac-nn",
    "brainiac-train",
    "brainiac-data",
    "brainiac",
    "brainiac-examples",
    "brainiac-benchmarks",
)
