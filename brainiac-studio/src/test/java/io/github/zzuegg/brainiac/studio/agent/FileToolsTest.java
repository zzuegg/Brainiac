package io.github.zzuegg.brainiac.studio.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.project.Project;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Reading by lines, editing, finding files and text, as agents use them on code. */
class FileToolsTest {

    @TempDir
    Path folder;

    private Project project;
    private final FileTools tools = new FileTools();

    @BeforeEach
    void setUp() throws Exception {
        project = new Project("demo", "Demo", "", folder, null, null, Map.of(), Map.of());
        write("src/main/java/demo/App.java", """
                package demo;

                public class App {
                    public static void main(String[] args) {
                        System.out.println("hello");
                    }
                }
                """);
        write("src/test/java/demo/AppTest.java", "class AppTest {\n    // TODO test main\n}\n");
        write("README.md", "# Demo\n\nSay hello.\n");
        write("build/classes/App.class", "hello");
        write("notes/ignored.txt", "hello");
        write(".gitignore", "notes/\n");
    }

    private void write(String path, String text) throws Exception {
        Path f = folder.resolve(path);
        Files.createDirectories(f.getParent());
        Files.writeString(f, text, StandardCharsets.UTF_8);
    }

    private <T> T in(java.util.concurrent.Callable<T> c) throws Exception {
        return ScopedValue.where(ChatService.SCOPE, project).call(c::call);
    }

    @Test
    void readsByLinesWithNumbers() throws Exception {
        String all = in(() -> tools.readFile("src/main/java/demo/App.java", Optional.empty(), Optional.empty()));
        assertTrue(all.startsWith("[src/main/java/demo/App.java: lines 1-7 of 7]"), all);
        assertTrue(all.contains("3\tpublic class App {"), all);
        String part = in(() -> tools.readFile("src/main/java/demo/App.java", Optional.of(4), Optional.of(2)));
        assertTrue(part.contains("lines 4-5 of 7; read on with startLine 6"), part);
        assertTrue(part.contains("5\t        System.out.println"), part);
        IllegalArgumentException missing = assertThrows(IllegalArgumentException.class,
                () -> in(() -> tools.readFile("src/main/java/demo/Ap.java", Optional.empty(), Optional.empty())));
        assertTrue(missing.getMessage().contains("App.java"), "names the file that is there: " + missing.getMessage());
    }

    @Test
    void editsShowTheChangedLinesAndReplaceAll() throws Exception {
        String r = in(() -> tools.editFile("src/main/java/demo/App.java", "\"hello\"", "\"hi\"", Optional.empty()));
        assertTrue(r.startsWith("changed src/main/java/demo/App.java at line 5"), r);
        assertTrue(r.contains("5\t        System.out.println(\"hi\");"), r);
        write("x.txt", "a b a b a\n");
        IllegalArgumentException twice = assertThrows(IllegalArgumentException.class,
                () -> in(() -> tools.editFile("x.txt", "a", "c", Optional.empty())));
        assertTrue(twice.getMessage().contains("3 times"), twice.getMessage());
        in(() -> tools.editFile("x.txt", "a", "c", Optional.of(true)));
        assertEquals("c b c b c\n", Files.readString(folder.resolve("x.txt")));
        IllegalArgumentException close = assertThrows(IllegalArgumentException.class,
                () -> in(() -> tools.editFile("src/main/java/demo/App.java", "public class Ap {", "x", Optional.empty())));
        assertTrue(close.getMessage().contains("closest line is 3"), close.getMessage());
    }

    @Test
    void multiEditIsAllOrNone() throws Exception {
        String before = Files.readString(folder.resolve("src/main/java/demo/App.java"));
        assertThrows(IllegalArgumentException.class, () -> in(() -> tools.multiEdit("src/main/java/demo/App.java",
                List.of(new FileTools.Change("App", "Main", null), new FileTools.Change("missing", "x", null)))));
        assertEquals(before, Files.readString(folder.resolve("src/main/java/demo/App.java")), "nothing changed");
        in(() -> tools.multiEdit("src/main/java/demo/App.java", List.of(new FileTools.Change("class App", "class Main",
                null), new FileTools.Change("\"hello\"", "\"hi\"", false))));
        String after = Files.readString(folder.resolve("src/main/java/demo/App.java"));
        assertTrue(after.contains("class Main") && after.contains("\"hi\""), after);
    }

    @Test
    void movesFiles() throws Exception {
        in(() -> tools.moveFile("README.md", "docs/README.md"));
        assertFalse(Files.exists(folder.resolve("README.md")));
        assertTrue(Files.exists(folder.resolve("docs/README.md")));
        assertThrows(IllegalArgumentException.class, () -> in(() -> tools.moveFile("docs/README.md", ".gitignore")));
    }

    @Test
    void globSkipsBuildOutputAndIgnoredFolders() throws Exception {
        Map<String, Object> java = in(() -> tools.glob("**/*.java", Optional.empty(), Optional.empty()));
        assertEquals(2, java.get("total"), String.valueOf(java));
        Map<String, Object> any = in(() -> tools.glob("*", Optional.empty(), Optional.empty()));
        String files = String.valueOf(any.get("files"));
        assertFalse(files.contains("build/") || files.contains("notes/"), files);
        assertTrue(files.contains("README.md"), files);
        Map<String, Object> top = in(() -> tools.glob("src/**/App*.java", Optional.empty(), Optional.empty()));
        assertEquals(2, top.get("total"), String.valueOf(top));
    }

    @Test
    void grepFindsWithContextFilesAndCounts() throws Exception {
        Map<String, Object> lines = in(() -> tools.grep("HELLO", Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.of(1), Optional.empty(), Optional.empty(), Optional.empty()));
        assertEquals(2, lines.get("total"), "App.java and README.md, not build/ or notes/: " + lines);
        String s = String.valueOf(lines.get("matches"));
        assertTrue(s.contains("line=5") && s.contains("before="), s);
        Map<String, Object> exact = in(() -> tools.grep("HELLO", Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.of(true), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty()));
        assertEquals(0, exact.get("total"));
        Map<String, Object> javaFiles = in(() -> tools.grep("class \\w+", Optional.empty(), Optional.of("java"),
                Optional.of(true), Optional.empty(), Optional.empty(), Optional.of("files"), Optional.empty(),
                Optional.empty()));
        assertEquals(List.of("src/main/java/demo/App.java", "src/test/java/demo/AppTest.java"), javaFiles.get("files"));
        assertThrows(IllegalArgumentException.class, () -> in(() -> tools.grep("(", Optional.empty(), Optional.empty(),
                Optional.of(true), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty())));
    }

    @Test
    void treeShowsFoldersWithCounts() throws Exception {
        String t = in(() -> tools.tree(Optional.empty(), Optional.of(2)));
        assertTrue(t.startsWith("./  (4 files"), t);
        assertTrue(t.contains("  src/  (2 files"), t);
        assertFalse(t.contains("build/") || t.contains("notes/"), t);
        assertTrue(t.contains("  README.md"), t);
    }

    @Test
    void oldNamesReachTheRenamedTools() {
        assertEquals("read_file", ToolNames.current("read_project_file"));
        assertEquals("grep", ToolNames.current("search_files"));
        assertEquals("write_file", ToolNames.current("write_file"));
        Project limited = new Project("p", "P", "", folder, List.of("read_project_file"), null, Map.of(), Map.of());
        assertTrue(limited.allowsTool("read_file"));
    }

    @Test
    void theProjectsGuideIsReadForAgents() throws Exception {
        assertEquals("", AgentRuns.projectInstructions(project));
        write("CLAUDE.md", "Use tabs.");
        assertTrue(AgentRuns.projectInstructions(project).contains("Use tabs."));
        write("AGENTS.md", "Run ./gradlew test.\n" + "x".repeat(AgentRuns.GUIDE_CHARS));
        String g = AgentRuns.projectInstructions(project);
        assertTrue(g.startsWith("From AGENTS.md:") && g.contains("./gradlew test") && g.endsWith("for the rest]"), g);
    }
}
