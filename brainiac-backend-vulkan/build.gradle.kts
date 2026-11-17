plugins {
    id("brainiac.java-library")
}

// LWJGL's natives jar for the platform the build runs on (one module, org.lwjgl.natives,
// per platform, so only one can be on the module path).
val lwjglNatives: String = run {
    val os = System.getProperty("os.name").lowercase()
    val arm = System.getProperty("os.arch").let { it == "aarch64" || it == "arm64" }
    when {
        os.contains("windows") -> if (arm) "natives-windows-arm64" else "natives-windows"
        os.contains("mac") -> if (arm) "natives-macos-arm64" else "natives-macos"
        else -> if (arm) "natives-linux-arm64" else "natives-linux"
    }
}

dependencies {
    api(project(":brainiac-tensor"))
    // LWJGL is the one runtime dependency allowed: Vulkan bindings, no other libraries
    implementation(libs.lwjgl)
    implementation(libs.lwjgl.vulkan)
    implementation(variantOf(libs.lwjgl) { classifier(lwjglNatives) })
    testImplementation(project(":brainiac-nn"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}

// The tests compare every operation with the CPU backend. They are skipped where no Vulkan
// driver is installed; CI installs lavapipe (Mesa's Vulkan driver for the CPU), which the
// backend only uses when asked to consider CPU devices.
tasks.test {
    jvmArgs(
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules", "jdk.incubator.vector",
        "-Dbrainiac.vulkan.devices=all",
    )
    // -Pbrainiac.vulkan.validate=true runs the tests under the Khronos validation layer
    if (providers.gradleProperty("brainiac.vulkan.validate").orNull == "true") {
        jvmArgs("-Dbrainiac.vulkan.validate=true")
    }
}

// The shaders are compiled to SPIR-V ahead of time and kept in src/main/resources, so
// building needs no shader compiler. After changing src/main/glsl, run
// ./gradlew :brainiac-backend-vulkan:compileShaders (needs glslangValidator on the PATH).
abstract class CompileShaders @Inject constructor(private val exec: ExecOperations) : DefaultTask() {
    @get:InputDirectory
    abstract val sources: DirectoryProperty

    @get:OutputDirectory
    abstract val output: DirectoryProperty

    @TaskAction
    fun compile() {
        // matmul is compiled twice: 64 x 64 and 32 x 32 output tiles
        val variants = mapOf("matmul" to listOf("matmul64" to listOf("-DW=4"), "matmul32" to listOf("-DW=2")))
        val dir = sources.get().asFile
        dir.listFiles { f -> f.name.endsWith(".comp") }!!.sorted().forEach { src ->
            val base = src.name.removeSuffix(".comp")
            for ((name, defines) in variants[base] ?: listOf(base to emptyList())) {
                exec.exec {
                    workingDir = dir
                    commandLine(listOf("glslangValidator", "-V", "--target-env", "vulkan1.2") + defines
                            + listOf("-o", output.file("$name.spv").get().asFile.absolutePath, src.name))
                }
            }
        }
    }
}

tasks.register<CompileShaders>("compileShaders") {
    group = "build"
    description = "Compiles the GLSL compute shaders to SPIR-V (needs glslangValidator)."
    sources = layout.projectDirectory.dir("src/main/glsl")
    output = layout.projectDirectory.dir("src/main/resources/io/github/zzuegg/brainiac/backend/vulkan/shaders")
}
