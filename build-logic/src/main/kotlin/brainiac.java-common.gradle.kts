// Settings shared by every Brainiac module: Java toolchain, compiler flags, tests.

plugins {
    java
}

val libs = versionCatalogs.named("libs")
val javaVersion = providers.gradleProperty("brainiac.javaVersion").map(String::toInt).getOrElse(27)

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(javaVersion)
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
    options.release = javaVersion
    // -parameters keeps parameter names in the class files, so tools such as the studio's
    // builder can show "filters" and "kernel" instead of arg0 and arg1
    options.compilerArgs.addAll(listOf("-Xlint:all", "-Werror", "-parameters"))
}

dependencies {
    testImplementation(platform(libs.findLibrary("junit-bom").get()))
    testImplementation(libs.findLibrary("junit-jupiter").get())
    testRuntimeOnly(libs.findLibrary("junit-platform-launcher").get())
}

// Preloads the JVM's libjsig on Linux. A driver built on LLVM (such as the CPU drivers CI uses)
// installs its own SIGSEGV handler, but the JVM faults on purpose at every safepoint poll:
// without chaining, about one test run in three died with SIGSEGV and no crash report.
tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    doFirst {
        val test = this as Test
        val jsig = File(test.javaLauncher.get().metadata.installationPath.asFile, "lib/libjsig.so")
        if (System.getProperty("os.name").startsWith("Linux") && jsig.isFile
                && !test.environment.containsKey("LD_PRELOAD")) {
            test.environment("LD_PRELOAD", jsig.absolutePath)
        }
        // LWJGL extracts its natives to one shared folder by default; test JVMs of several
        // modules starting together then rewrote a library another one was loading (SIGBUS in
        // dlopen). Each test task extracts into its own folder.
        test.systemProperty("org.lwjgl.system.SharedLibraryExtractPath", File(test.temporaryDir, "lwjgl").absolutePath)
        // NVIDIA's Linux driver starts a "threaded optimizations" worker that crashed every JVM at
        // exit (SIGSEGV in libEGL_nvidia, after all its work was done) once Vulkan had been loaded,
        // even by listing devices; the driver reads the variable when it loads, before the JVM runs
        if (System.getProperty("os.name").startsWith("Linux") && !test.environment.containsKey("__GL_THREADED_OPTIMIZATIONS")
                && System.getenv("__GL_THREADED_OPTIMIZATIONS") == null) {
            test.environment("__GL_THREADED_OPTIMIZATIONS", "0")
        }
    }
}

tasks.withType<JavaExec>().configureEach {
    doFirst {
        val exec = this as JavaExec
        val launcher = exec.javaLauncher.orNull
        val jsig = launcher?.let { File(it.metadata.installationPath.asFile, "lib/libjsig.so") }
        if (System.getProperty("os.name").startsWith("Linux") && jsig != null && jsig.isFile
                && !exec.environment.containsKey("LD_PRELOAD")) {
            exec.environment("LD_PRELOAD", jsig.absolutePath)
        }
        // as for tests: NVIDIA's threaded-optimization worker crashed JVMs at exit
        if (System.getProperty("os.name").startsWith("Linux") && !exec.environment.containsKey("__GL_THREADED_OPTIMIZATIONS")
                && System.getenv("__GL_THREADED_OPTIMIZATIONS") == null) {
            exec.environment("__GL_THREADED_OPTIMIZATIONS", "0")
        }
    }
}
