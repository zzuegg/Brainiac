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
    "brainiac-backend-vulkan",
    "brainiac-autograd",
    "brainiac-nn",
    "brainiac-train",
    "brainiac-data",
    "brainiac-io",
    "brainiac-llm",
    "brainiac-diffusion",
    "brainiac-rl",
    "brainiac-evolve",
    "brainiac",
    "brainiac-examples",
    "brainiac-benchmarks",
    "brainiac-experiments",
    "brainiac-studio",
)
