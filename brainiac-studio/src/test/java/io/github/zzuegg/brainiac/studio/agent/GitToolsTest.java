package io.github.zzuegg.brainiac.studio.agent;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.project.Project;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The git tools on a real repository: show, blame, restore, log filters and merges with conflicts. */
class GitToolsTest {

    @TempDir
    Path folder;

    @TempDir
    Path data;

    private Project project;
    private CodeTools tools;

    @BeforeEach
    void setUp() throws Exception {
        assumeTrue(git("--version"), "git is installed");
        project = new Project("demo", "Demo", "", folder, null, null, Map.of(), Map.of());
        tools = new CodeTools(new Services(data.resolve("services.json"), null));
        git("init", "-q", "-b", "main");
        git("config", "user.email", "test@example.com");
        git("config", "user.name", "Tester");
        Files.writeString(folder.resolve("a.txt"), "one\ntwo\n");
        git("add", "-A");
        git("commit", "-q", "-m", "First lines");
    }

    private boolean git(String... args) throws Exception {
        List<String> cmd = new java.util.ArrayList<>(List.of("git", "-C", folder.toString()));
        cmd.addAll(List.of(args));
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        p.getInputStream().readAllBytes();
        return p.waitFor(60, TimeUnit.SECONDS) && p.exitValue() == 0;
    }

    private <T> T in(java.util.concurrent.Callable<T> c) throws Exception {
        return ScopedValue.where(ChatService.SCOPE, project).call(c::call);
    }

    @Test
    void showsBlamesAndFiltersTheLog() throws Exception {
        String show = in(() -> tools.gitShow(Optional.empty(), "HEAD", Optional.empty()));
        assertTrue(show.contains("First lines") && show.contains("+two"), show);
        String old = in(() -> tools.gitShow(Optional.empty(), "HEAD", Optional.of("a.txt")));
        assertEquals("one\ntwo\n", old);
        String blame = in(() -> tools.gitBlame(Optional.empty(), "a.txt", Optional.of(2), Optional.empty()));
        assertTrue(blame.contains("Tester") && blame.contains("two") && !blame.contains("one"), blame);
        String log = in(() -> tools.gitLog(Optional.empty(), Optional.empty(), Optional.empty(), Optional.of("first"),
                Optional.empty(), Optional.empty()));
        assertTrue(log.contains("First lines"), log);
        String none = in(() -> tools.gitLog(Optional.empty(), Optional.empty(), Optional.empty(), Optional.of("nothing"),
                Optional.empty(), Optional.empty()));
        assertEquals("(nothing)", none);
    }

    @Test
    void restorePutsFilesBack() throws Exception {
        Files.writeString(folder.resolve("a.txt"), "changed\n");
        String r = in(() -> tools.gitRestore(Optional.empty(), List.of("a.txt"), Optional.empty()));
        assertTrue(r.startsWith("put back 1 file(s)"), r);
        assertEquals("one\ntwo\n", Files.readString(folder.resolve("a.txt")));
        String same = in(() -> tools.gitRestore(Optional.empty(), List.of("a.txt"), Optional.empty()));
        assertTrue(same.startsWith("nothing differed"), same);
    }

    @Test
    void mergesAndReportsConflicts() throws Exception {
        git("switch", "-q", "-c", "side");
        Files.writeString(folder.resolve("a.txt"), "one\nside\n");
        git("commit", "-q", "-am", "Side");
        git("switch", "-q", "main");
        Files.writeString(folder.resolve("b.txt"), "b\n");
        git("add", "-A");
        git("commit", "-q", "-m", "Main");
        String clean = in(() -> tools.gitMerge(Optional.empty(), Optional.of("side"), Optional.empty()));
        assertEquals("one\nside\n", Files.readString(folder.resolve("a.txt")), clean);

        git("switch", "-q", "-c", "other", "HEAD~1");
        Files.writeString(folder.resolve("a.txt"), "one\nother\n");
        git("commit", "-q", "-am", "Other");
        git("switch", "-q", "main");
        String conflict = in(() -> tools.gitMerge(Optional.empty(), Optional.of("other"), Optional.empty()));
        assertTrue(conflict.startsWith("the merge stopped") && conflict.contains("a.txt"), conflict);
        assertTrue(Files.readString(folder.resolve("a.txt")).contains("<<<<<<<"));
        in(() -> tools.gitMerge(Optional.empty(), Optional.empty(), Optional.of(true)));
        assertEquals("one\nside\n", Files.readString(folder.resolve("a.txt")));
    }
}
