package io.github.zzuegg.brainiac.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/** JSON Lines: one JSON value per line, the format of most text and chat datasets. */
public final class Jsonl {

    private Jsonl() {
    }

    /**
     * Reads every line's object.
     *
     * @param file the file
     * @return one map per non-blank line
     * @throws IOException if it cannot be read or a line is not a JSON object
     */
    public static List<Map<String, Object>> read(Path file) throws IOException {
        try (Stream<Map<String, Object>> s = stream(file)) {
            return s.toList();
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
    }

    /**
     * Streams the objects without holding the file, for datasets larger than memory. Close
     * the stream.
     *
     * @param file the file
     * @return one map per non-blank line
     * @throws IOException if it cannot be opened
     */
    public static Stream<Map<String, Object>> stream(Path file) throws IOException {
        BufferedReader r = Files.newBufferedReader(file, StandardCharsets.UTF_8);
        int[] line = {0};
        return r.lines().map(l -> {
            line[0]++;
            return l;
        }).filter(l -> !l.isBlank()).map(l -> {
            try {
                return Json.object(Json.parse(l));
            } catch (RuntimeException e) {
                throw new UncheckedIOException(new IOException(file + " line " + line[0] + ": " + e.getMessage(), e));
            }
        }).onClose(() -> {
            try {
                r.close();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        });
    }

    /**
     * Reads one text field of every line, such as {@code "text"} of a pretraining corpus.
     *
     * @param file the file
     * @param field the field
     * @return the values, skipping lines without the field
     * @throws IOException if it cannot be read
     */
    public static List<String> texts(Path file, String field) throws IOException {
        try (Stream<Map<String, Object>> s = stream(file)) {
            return s.map(o -> o.get(field)).filter(v -> v instanceof String).map(v -> (String) v).toList();
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
    }

    /**
     * Writes values, one per line.
     *
     * @param file where to write
     * @param values JSON values: maps, lists, strings, numbers, booleans
     * @throws IOException if it cannot be written
     */
    public static void write(Path file, List<?> values) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            for (Object v : values) {
                w.write(Json.write(v));
                w.write('\n');
            }
        }
    }
}
