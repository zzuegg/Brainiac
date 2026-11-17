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
    // LWJGL is the one runtime dependency allowed: Vulkan bindings and the shaderc
    // shader compiler, no other libraries
    implementation(libs.lwjgl)
    implementation(libs.lwjgl.vulkan)
    implementation(variantOf(libs.lwjgl) { classifier(lwjglNatives) })
    // shaderc compiles the shaders generated for fused element-wise chains at run time
    implementation(libs.lwjgl.shaderc)
    implementation(variantOf(libs.lwjgl.shaderc) { classifier(lwjglNatives) })
    testImplementation(project(":brainiac-nn"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}

// The shared GLSL is also read at run time, as the prelude of the shaders generated for fused
// element-wise chains
tasks.processResources {
    from("src/main/glsl") {
        include("*.glsl")
        into("io/github/zzuegg/brainiac/backend/vulkan/shaders")
    }
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
    // -Pbench prints matrix product speeds (MatmulBenchmarkTest)
    if (providers.gradleProperty("bench").isPresent) {
        jvmArgs("-Dbrainiac.bench=true")
        testLogging.showStandardStreams = true
    }
    // -Pbrainiac.vulkan.profile=true runs the tests with every dispatch timed (it must not change results)
    if (providers.gradleProperty("brainiac.vulkan.profile").orNull == "true") {
        jvmArgs("-Dbrainiac.vulkan.profile=true")
    }
    // -Pbrainiac.vulkan.diagnose=<case> compiles test shaders one by one (DriverDiagnosisTest)
    providers.gradleProperty("brainiac.vulkan.diagnose").orNull?.let {
        jvmArgs("-Dbrainiac.vulkan.diagnose=$it")
        testLogging.showStandardStreams = true
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
        // every shader once per number format (random is float32 only); matmul twice more:
        // 64 x 64 and 32 x 32 output tiles; matmul_packed for 1, 2, 4 and 8 rows
        val types = listOf("f32", "f64", "f16", "bf16")
        val variants = mapOf(
            "matmul" to listOf("matmul64" to listOf("-DW=4"), "matmul32" to listOf("-DW=2"),
                // operands read through strides: a transposed matrix in place
                "matmul64s" to listOf("-DW=4", "-DSTRIDED"), "matmul32s" to listOf("-DW=2", "-DSTRIDED")),
            "matmul_packed" to listOf(1, 2, 4, 8).map { "matmul_packed$it" to listOf("-DROWS=$it") },
            "matmul_vec" to listOf(1, 2, 4, 8).map { "matmul_vec$it" to listOf("-DROWS=$it") },
            // register arrays sized for the head: 32 keeps more work groups resident
            "attention_dq" to listOf("attention_dq" to emptyList(), "attention_dq32" to listOf("-DATT_D=32")),
            "attention_dkv" to listOf("attention_dkv" to emptyList(), "attention_dkv32" to listOf("-DATT_D=32")),
            // flash decoding: the keys of a row shared out among work groups (attention_combine joins them)
            "attention" to listOf("attention" to emptyList(), "attention_split" to listOf("-DSPLIT")),
        )
        val dir = sources.get().asFile
        output.get().asFile.listFiles { f -> f.name.endsWith(".spv") }?.forEach { it.delete() }
        dir.listFiles { f -> f.name.endsWith(".comp") }!!.sorted().forEach { src ->
            val base = src.name.removeSuffix(".comp")
            for ((name, defines) in variants[base] ?: listOf(base to emptyList())) {
                // cooperative matrices: float16 only
                for (type in if (base == "random") listOf("") else if (base == "matmul_coop") listOf("f16") else types) {
                    val file = if (type.isEmpty()) "$name.spv" else "${name}_$type.spv"
                    val typeDefine = if (type.isEmpty()) emptyList() else listOf("-DTYPE_" + type.uppercase())
                    exec.exec {
                        workingDir = dir
                        commandLine(listOf("glslangValidator", "-V", "--target-env", "vulkan1.2") + defines
                                + typeDefine + listOf("-o", output.file(file).get().asFile.absolutePath, src.name))
                    }
                    // float64 unary and activation once more per operation: AMD's LLPC compiler
                    // crashes on the switch over all of them (counts: UnaryOp, ActivationOp); optimized, so
                    // the switch is gone from the SPIR-V
                    val ops = mapOf("unary" to 16, "activation" to 15)[base]
                    if (type == "f64" && ops != null) {
                        for (op in 0 until ops) {
                            exec.exec {
                                workingDir = dir
                                commandLine(listOf("glslangValidator", "-V", "--target-env", "vulkan1.2", "-Os", "-DOP=$op")
                                        + typeDefine + listOf("-o",
                                        output.file("${name}_${type}_$op.spv").get().asFile.absolutePath, src.name))
                            }
                        }
                    }
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
