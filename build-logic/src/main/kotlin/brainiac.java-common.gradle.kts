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
    options.compilerArgs.addAll(listOf("-Xlint:all", "-Werror"))
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
    }
}
