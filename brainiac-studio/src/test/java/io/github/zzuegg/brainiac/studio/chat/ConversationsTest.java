package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Conversations kept as files: saved, listed newest first, finished, deleted, and kept safe. */
class ConversationsTest {

    @TempDir
    Path dir;

    private static Map<String, Object> convo(String id, long updated) {
        Map<String, Object> c = new HashMap<>();
        c.put("id", id);
        c.put("title", "about " + id);
        c.put("messages", List.of(Map.of("role", "user", "text", "hi")));
        c.put("updated", updated);
        return c;
    }

    @Test
    void savedConversationsComeBackNewestFirst() throws IOException {
        Conversations store = new Conversations(dir.resolve("chats"));
        assertEquals(List.of(), store.list());
        store.save(convo("aaaa1", 10));
        store.save(convo("bbbb2", 30));
        store.save(convo("cccc3", 20));
        assertEquals(List.of("bbbb2", "cccc3", "aaaa1"), store.list().stream().map(c -> c.get("id")).toList());
        assertEquals("hi", ((Map<?, ?>) ((List<?>) store.list().getFirst().get("messages")).getFirst()).get("text"));
    }

    @Test
    void savingAgainReplacesAndKeepsTheStatus() throws IOException {
        Conversations store = new Conversations(dir);
        Map<String, Object> c = convo("abcd", 1);
        store.save(c);
        c.put("status", "finished");
        c.put("finishedAt", 5L);
        store.save(c);
        List<Map<String, Object>> all = store.list();
        assertEquals(1, all.size());
        assertEquals("finished", all.getFirst().get("status"));
        assertTrue(store.delete("abcd"));
        assertFalse(store.delete("abcd"));
        assertEquals(List.of(), store.list());
    }

    @Test
    void badConversationsAreRefused() throws IOException {
        Conversations store = new Conversations(dir);
        assertThrows(IllegalArgumentException.class, () -> store.save(convo("../../keys", 1)));
        assertThrows(IllegalArgumentException.class, () -> store.save(convo("AB", 1)));
        assertThrows(IllegalArgumentException.class, () -> store.delete("../x"));
        Map<String, Object> noTitle = convo("abcd", 1);
        noTitle.remove("title");
        assertThrows(IllegalArgumentException.class, () -> store.save(noTitle));
        Map<String, Object> noMessages = convo("abcd", 1);
        noMessages.put("messages", "hi");
        assertThrows(IllegalArgumentException.class, () -> store.save(noMessages));
        Map<String, Object> odd = convo("abcd", 1);
        odd.put("status", "archived");
        assertThrows(IllegalArgumentException.class, () -> store.save(odd));
        assertFalse(Files.exists(dir.resolve("abcd.json")));
    }

    @Test
    void aBrokenFileIsSkippedNotDeleted() throws IOException {
        Conversations store = new Conversations(dir);
        store.save(convo("good1", 1));
        Files.writeString(dir.resolve("bad01.json"), "{not json");
        assertEquals(List.of("good1"), store.list().stream().map(c -> c.get("id")).toList());
        assertTrue(Files.exists(dir.resolve("bad01.json")));
    }

    @Test
    void theOldestGoBeyondTheLimit() throws IOException {
        Conversations store = new Conversations(dir, 2);
        store.save(convo("old01", 1));
        store.save(convo("mid02", 2));
        Files.setLastModifiedTime(dir.resolve("old01.json"), FileTime.fromMillis(1_000));
        Files.setLastModifiedTime(dir.resolve("mid02.json"), FileTime.fromMillis(2_000));
        store.save(convo("new03", 3));
        assertEquals(List.of("new03", "mid02"), store.list().stream().map(c -> c.get("id")).toList());
    }
}
