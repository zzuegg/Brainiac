package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** What the page remembers: kept in a file, changed by name, safe to put in the page. */
class PageStateTest {

    @TempDir
    Path dir;

    @Test
    void valuesSurviveARestartAndNullRemovesOne() throws IOException {
        Path f = dir.resolve("data").resolve("page.json");
        PageState s = new PageState(f);
        assertEquals(Map.of(), s.values());
        s.merge(Map.of("layout", "ultra", "chatTabs", List.of("abcd", "efgh"), "scale", 1.25));
        Map<String, Object> gone = new HashMap<>();
        gone.put("layout", null);
        s.merge(gone);
        PageState again = new PageState(f);
        assertEquals(Map.of("chatTabs", List.of("abcd", "efgh"), "scale", 1.25), again.values());
    }

    @Test
    void namesAndSizeAreChecked() throws IOException {
        PageState s = new PageState(dir.resolve("page.json"));
        assertThrows(IllegalArgumentException.class, () -> s.merge(Map.of("../x", 1)));
        assertThrows(IllegalArgumentException.class, () -> s.merge(Map.of("", 1)));
        assertThrows(IllegalArgumentException.class, () -> s.merge(Map.of("big", "x".repeat(PageState.MAX_CHARS))));
        Map<String, Object> many = new HashMap<>();
        for (int i = 0; i <= PageState.MAX_VALUES; i++) {
            many.put("k" + i, i);
        }
        assertThrows(IllegalArgumentException.class, () -> s.merge(many));
        assertEquals(Map.of(), s.values());
        assertFalse(Files.exists(dir.resolve("page.json")));
    }

    @Test
    void theJsonCannotEndTheScriptItSitsIn() throws IOException {
        PageState s = new PageState(dir.resolve("page.json"));
        s.merge(Map.of("search", "</script><script>alert(1)</script>"));
        assertFalse(s.json().contains("<"));
        assertTrue(s.json().contains("\\u003c/script>"));
    }

    @Test
    void aBrokenFileStartsEmptyAndIsReplacedOnTheNextChange() throws IOException {
        Path f = dir.resolve("page.json");
        Files.writeString(f, "{broken");
        PageState s = new PageState(f);
        assertEquals(Map.of(), s.values());
        s.merge(Map.of("density", "compact"));
        assertEquals(Map.of("density", "compact"), new PageState(f).values());
    }
}
