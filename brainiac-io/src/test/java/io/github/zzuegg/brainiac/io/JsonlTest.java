package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class JsonlTest {

    @Test
    void readsWritesAndReportsBadLines(@TempDir Path dir) throws IOException {
        Path f = dir.resolve("a.jsonl");
        Jsonl.write(f, List.of(Map.of("text", "hello"), Map.of("text", "world", "n", 2), Map.of("other", true)));
        assertEquals(3, Jsonl.read(f).size());
        assertEquals(List.of("hello", "world"), Jsonl.texts(f, "text"));
        Files.writeString(f, "{\"a\": 1}\n\n[1, 2\n");
        IOException e = assertThrows(IOException.class, () -> Jsonl.read(f));
        assertEquals(true, e.getMessage().contains("line 3"), e.getMessage());
    }
}
