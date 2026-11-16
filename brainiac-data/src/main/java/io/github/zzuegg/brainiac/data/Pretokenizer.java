package io.github.zzuegg.brainiac.data;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Helpers the tokenizers share: cutting text into words, and byte strings as map keys. */
final class Pretokenizer {

    /** Words with their leading space, numbers, punctuation runs, and spaces, as in GPT-2. */
    private static final Pattern CHUNKS = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    static final String HEADER = "brainiac-tokenizer ";

    private Pretokenizer() {
    }

    /** The chunks of a text, in order; tokens never cross a chunk boundary. */
    static List<String> chunks(String text) {
        List<String> out = new ArrayList<>();
        Matcher m = CHUNKS.matcher(text);
        while (m.find()) {
            out.add(m.group());
        }
        return out;
    }

    /** Each distinct chunk with how often it occurs, in order of first appearance. */
    static Map<String, Integer> counts(String text) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (String chunk : chunks(text)) {
            counts.merge(chunk, 1, Integer::sum);
        }
        return counts;
    }

    static byte[] utf8(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    /** Bytes as a string with one char per byte, so byte strings can key a hash map. */
    static String key(byte[] b, int from, int to) {
        return new String(b, from, to - from, StandardCharsets.ISO_8859_1);
    }

    static String key(byte[] b) {
        return key(b, 0, b.length);
    }

    static byte[] unkey(String key) {
        return key.getBytes(StandardCharsets.ISO_8859_1);
    }

    static String hex(byte[] b) {
        return HexFormat.of().formatHex(b);
    }

    static byte[] unhex(String s) {
        return HexFormat.of().parseHex(s);
    }

    static void write(Path path, String kind, List<String> lines) throws IOException {
        StringBuilder b = new StringBuilder(HEADER).append(kind).append(' ').append(lines.size()).append('\n');
        for (String line : lines) {
            b.append(line).append('\n');
        }
        Files.writeString(path, b, StandardCharsets.UTF_8);
    }

    /** The lines after the header, checked against the count the header gives. */
    static List<String> body(List<String> lines, Path path) throws IOException {
        String[] head = lines.get(0).split(" ");
        int count = Integer.parseInt(head[2]);
        if (lines.size() < count + 1) {
            throw new IOException(path + " should have " + count + " entries, has " + (lines.size() - 1));
        }
        return lines.subList(1, count + 1);
    }

    /** A vocabulary of 256 single bytes followed by longer pieces, as hex lines. */
    static List<String> hexLines(List<byte[]> pieces) {
        List<String> out = new ArrayList<>(pieces.size());
        for (byte[] p : pieces) {
            out.add(hex(p));
        }
        return out;
    }
}
