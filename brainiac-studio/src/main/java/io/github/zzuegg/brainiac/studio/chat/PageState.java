package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * What the page remembers between visits (open tabs and panes, layout, interface size, chosen
 * models, folded groups), kept in one JSON file in the studio's folder instead of the browser,
 * so it is the same in every browser on this PC. The page owns the values; the studio checks
 * only the names and the size.
 */
public final class PageState {

    private static final System.Logger LOG = System.getLogger(PageState.class.getName());
    private static final Pattern NAME = Pattern.compile("[A-Za-z][A-Za-z0-9_.-]{0,63}");
    /** At most this many values are kept. */
    static final int MAX_VALUES = 200;
    /** At most this many characters of JSON are kept. */
    static final int MAX_CHARS = 1 << 20;

    private final Path file;
    private final Map<String, Object> values = new LinkedHashMap<>();

    /**
     * Reads the file; a missing or unreadable one starts empty (an unreadable one is reported
     * and left in place until the next change replaces it).
     *
     * @param file the file; it need not exist yet
     */
    public PageState(Path file) {
        this.file = file;
        if (Files.isRegularFile(file)) {
            try {
                values.putAll(Json.object(Json.parse(Files.readString(file, StandardCharsets.UTF_8))));
            } catch (IOException | IllegalArgumentException bad) {
                LOG.log(System.Logger.Level.WARNING, "cannot read " + file + ", starting empty: " + bad.getMessage());
            }
        }
    }

    /**
     * Returns every value.
     *
     * @return a copy of the values by name
     */
    public synchronized Map<String, Object> values() {
        return new LinkedHashMap<>(values);
    }

    /**
     * Returns every value as JSON that is safe to put inside an HTML {@code <script>} element.
     *
     * @return the JSON, with {@code <} written as {@code <}
     */
    public synchronized String json() {
        return Json.write(values).replace("<", "\\u003c");
    }

    /**
     * Changes some values and saves the file.
     *
     * @param changes values by name; null removes one
     * @throws IOException if the file cannot be written
     */
    public synchronized void merge(Map<String, Object> changes) throws IOException {
        Map<String, Object> next = new LinkedHashMap<>(values);
        for (Map.Entry<String, Object> e : changes.entrySet()) {
            if (!NAME.matcher(e.getKey()).matches()) {
                throw new IllegalArgumentException("not a setting name: " + e.getKey());
            }
            if (e.getValue() == null) {
                next.remove(e.getKey());
            } else {
                next.put(e.getKey(), e.getValue());
            }
        }
        if (next.size() > MAX_VALUES) {
            throw new IllegalArgumentException("at most " + MAX_VALUES + " settings are kept");
        }
        String text = Json.write(next);
        if (text.length() > MAX_CHARS) {
            throw new IllegalArgumentException("the page state is limited to " + MAX_CHARS + " characters");
        }
        Files.createDirectories(file.toAbsolutePath().getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.writeString(tmp, text, StandardCharsets.UTF_8);
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        values.clear();
        values.putAll(next);
    }
}
