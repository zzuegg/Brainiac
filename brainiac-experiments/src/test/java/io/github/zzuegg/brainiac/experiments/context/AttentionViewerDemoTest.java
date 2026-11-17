package io.github.zzuegg.brainiac.experiments.context;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AttentionViewerDemoTest {

    @Test
    void writesASelfContainedPageWithEveryStepAndEviction(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("cache.html");
        int steps = AttentionViewerDemo.write(file, 40, 8, false);
        String html = Files.readString(file);
        assertEquals(40, steps);
        assertTrue(html.startsWith("<!doctype html>"));
        assertTrue(html.endsWith("</html>\n"));
        assertTrue(!html.contains("<script") && !html.contains("http"), "no scripts or external files");
        // one row per step, one cell per (step, position)
        assertEquals(1 + 40, html.split("<tr><th>", -1).length - 1);   // header + rows
        assertEquals(40 * 40, html.split("<td", -1).length - 1);
        // 40 tokens through 8 slots: 32 dropped, each outlined once, then hatched
        assertEquals(32, html.split("class=\"gone drop\"|class=\"drop\"", -1).length - 1);
        assertTrue(html.contains("class=\"gone\""));
        // sinks: the first two tokens are never dropped and keep being read
        assertTrue(html.contains("title=\"Q") || html.contains("title=\"."));
    }
}
