package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.Test;

/** Files from {@code parquet/make_parquet.py} (pyarrow), against the rows pyarrow read back. */
class ParquetTest {

    private static Path resource(String name) throws Exception {
        return Path.of(ParquetTest.class.getResource("parquet/" + name).toURI());
    }

    @ParameterizedTest
    @ValueSource(strings = {"snappy.parquet", "zstd-v2.parquet", "gzip.parquet", "lz4.parquet", "plain.parquet",
            "delta.parquet"})
    void readsWhatPyarrowWrote(String file) throws Exception {
        List<Map<String, Object>> expected = Jsonl.read(resource("expected.jsonl"));
        List<Map<String, Object>> rows = Parquet.read(resource(file));
        assertEquals(expected.size(), rows.size());
        for (int i = 0; i < rows.size(); i++) {
            same(expected.get(i), rows.get(i), file + " row " + i);
        }
        try (Parquet p = Parquet.open(resource(file))) {
            assertEquals(List.of("id", "small", "score", "ratio", "flag", "text", "blob", "tags", "messages", "matrix",
                    "meta", "attrs"), p.columns());
            assertEquals(300, p.rows());
        }
        try (Stream<Map<String, Object>> s = Parquet.stream(resource(file))) {
            assertEquals(300, s.count());
        }
        assertEquals(expected.stream().map(r -> r.get("text")).toList(), Parquet.texts(resource(file), "text"));
    }

    @Test
    void refusesOtherFiles(@TempDir Path dir) throws Exception {
        Path junk = dir.resolve("x.parquet");
        Files.writeString(junk, "PAR1 not really PAR1");
        assertThrows(IllegalArgumentException.class, () -> Parquet.read(junk));
    }

    /** JSON numbers are Long or Double; binary is {"$bytes": hex}. */
    private static void same(Object expected, Object actual, String where) {
        if (expected == null || actual == null) {
            assertEquals(expected, actual, where);
        } else if (expected instanceof Map<?, ?> e && e.containsKey("$bytes")) {
            assertTrue(actual instanceof byte[], where + ": " + actual);
            assertEquals(e.get("$bytes"), HexFormat.of().formatHex((byte[]) actual), where);
        } else if (expected instanceof Map<?, ?> e) {
            Map<?, ?> a = (Map<?, ?>) actual;
            assertEquals(e.keySet(), a.keySet(), where);
            for (Object k : e.keySet()) {
                same(e.get(k), a.get(k), where + "." + k);
            }
        } else if (expected instanceof List<?> e) {
            List<?> a = (List<?>) actual;
            assertEquals(e.size(), a.size(), where + " " + a);
            for (int i = 0; i < e.size(); i++) {
                same(e.get(i), a.get(i), where + "[" + i + "]");
            }
        } else if (expected instanceof Number e) {
            if (!(actual instanceof Number a)) {
                fail(where + ": " + actual + " is not a number");
                return;
            }
            assertEquals(e.doubleValue(), a.doubleValue(), 0, where);
        } else {
            assertEquals(expected, actual, where);
        }
    }
}
