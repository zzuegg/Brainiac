package io.github.zzuegg.brainiac.studio.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Compile errors and test results read from what builds leave behind, and the commands that make them. */
class BuildOutputTest {

    @TempDir
    Path root;

    @TempDir
    Path copy;

    @Test
    void readsCompileErrorsOfSeveralTools() {
        String out = """
                > Task :app:compileJava FAILED
                /work/app/src/main/java/demo/App.java:12: error: cannot find symbol
                        foo();
                        ^
                  symbol:   method foo()
                  location: class App
                /work/app/src/main/java/demo/App.java:3: warning: [deprecation] Old in demo has been deprecated
                e: file:///work/lib/src/main/kotlin/Lib.kt:7:9 Unresolved reference: bar
                src/web/main.ts(4,2): error TS2322: Type 'string' is not assignable to type 'number'.
                error[E0425]: cannot find value `x` in this scope
                  --> src/main.rs:5:13
                /usr/lib/jvm/Other.java:1: error: outside the project
                BUILD FAILED in 2s
                """;
        List<Map<String, Object>> e = BuildOutput.errors(out, copy, root, "log");
        assertEquals(4, e.size(), String.valueOf(e));
        assertEquals("app/src/main/java/demo/App.java", e.get(0).get("file"));
        assertEquals(12, e.get(0).get("line"));
        assertTrue(String.valueOf(e.get(0).get("message")).contains("symbol:   method foo()"), String.valueOf(e.get(0)));
        assertEquals("lib/src/main/kotlin/Lib.kt", e.get(1).get("file"));
        assertEquals(9, e.get(1).get("column"));
        assertEquals("src/web/main.ts", e.get(2).get("file"));
        assertEquals("src/main.rs", e.get(3).get("file"));
        assertEquals("cannot find value `x` in this scope", e.get(3).get("message"));
        assertEquals(1, BuildOutput.warnings(out));
    }

    @Test
    void readsJUnitReportsWithTheFailingLine() throws Exception {
        Path src = root.resolve("app/src/test/java/demo/AppTest.java");
        Files.createDirectories(src.getParent());
        Files.writeString(src, "class AppTest {}");
        Path report = copy.resolve("app/build/test-results/test/TEST-demo.AppTest.xml");
        Files.createDirectories(report.getParent());
        Files.writeString(report, """
                <?xml version="1.0" encoding="UTF-8"?>
                <testsuite name="demo.AppTest" tests="3" skipped="1" failures="1" errors="0">
                  <testcase name="adds()" classname="demo.AppTest" time="0.01"/>
                  <testcase name="skips()" classname="demo.AppTest" time="0"><skipped/></testcase>
                  <testcase name="fails()" classname="demo.AppTest" time="0.02">
                    <failure message="expected: &lt;2&gt; but was: &lt;3&gt;" type="org.opentest4j.AssertionFailedError">org.opentest4j.AssertionFailedError: expected: &lt;2&gt; but was: &lt;3&gt;
                	at org.junit.jupiter.api.AssertionUtils.fail(AssertionUtils.java:151)
                	at demo.AppTest.fails(AppTest.java:21)
                	at java.base/java.lang.reflect.Method.invoke(Method.java:580)
                </failure>
                  </testcase>
                </testsuite>
                """);
        BuildOutput.Tests t = BuildOutput.tests(copy, root);
        assertTrue(t.found());
        assertEquals(1, t.passed());
        assertEquals(1, t.skipped());
        Map<String, Object> f = t.failed().getFirst();
        assertEquals("demo.AppTest.fails()", f.get("test"));
        assertEquals("app/src/test/java/demo/AppTest.java", f.get("file"));
        assertEquals(21, f.get("line"));
        assertFalse(String.valueOf(f.get("stack")).contains("org.junit"), String.valueOf(f.get("stack")));
        BuildOutput.clearReports(copy);
        assertFalse(BuildOutput.tests(copy, root).found());
    }

    @Test
    void picksTheBuildToolsCommands() throws Exception {
        Files.writeString(root.resolve("gradlew"), "#!/bin/sh");
        CodeTools.Tool g = CodeTools.Tool.detect(root);
        assertEquals("sh ./gradlew :brainiac-studio:test --console=plain --continue --tests '*FileToolsTest'",
                g.testCommand("brainiac-studio", "*FileToolsTest"));
        assertEquals("sh ./gradlew classes testClasses --console=plain --continue", g.buildCommand(""));
        assertThrows(IllegalArgumentException.class, () -> g.testCommand("", "x'; rm -rf /"));
        assertThrows(IllegalArgumentException.class, () -> g.buildCommand("a;b"));
        Path other = root.resolve("web");
        Files.createDirectories(other);
        Files.writeString(other.resolve("package.json"), "{}");
        assertEquals("npm test -- 'adds'", CodeTools.Tool.detect(other).testCommand("", "adds"));
        assertThrows(IllegalArgumentException.class, () -> CodeTools.Tool.detect(root.resolve("web").resolveSibling("none")));
    }

    @Test
    void theSandboxKeepsBuildOutputOnItsSide() throws Exception {
        Files.writeString(root.resolve("A.java"), "class A {}");
        Files.createDirectories(root.resolve("build"));
        Files.writeString(root.resolve("build/old.class"), "x");
        Path work = copy.resolve("work");
        Map<String, Long> first = CodeTools.sync(root, work);
        assertEquals(List.of("A.java"), List.copyOf(first.keySet()));
        assertFalse(Files.exists(work.resolve("build/old.class")), "the project's own build output stays out");
        Files.createDirectories(work.resolve("build"));
        Files.writeString(work.resolve("build/A.class"), "compiled");
        Files.delete(root.resolve("A.java"));
        Files.writeString(root.resolve("B.java"), "class B {}");
        CodeTools.sync(root, work);
        assertFalse(Files.exists(work.resolve("A.java")), "removed from the project, removed from the copy");
        assertTrue(Files.exists(work.resolve("B.java")));
        assertTrue(Files.exists(work.resolve("build/A.class")), "build output is kept for the next build");
    }
}
