package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * CLIP's byte-level BPE tokenizer: lower-cased words split by a regular expression, each
 * word's last symbol marked with {@code </w>}, merged pair by pair in rank order. Reads a
 * diffusers {@code tokenizer} folder ({@code vocab.json} and {@code merges.txt}, or
 * {@code tokenizer.json}).
 */
public final class ClipTokenizer {

    private static final Pattern WORDS = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);

    private static final String[] BYTE_CHARS = byteChars();

    private final Map<String, Integer> vocab;
    private final Map<String, Integer> ranks;
    private final int bos;
    private final int eos;
    private final int pad;
    private final String padText;
    private final int length;
    private final Map<String, int[]> cache = new HashMap<>();

    /**
     * Creates a tokenizer from its tables.
     *
     * @param vocab token to id
     * @param merges the merges, best first, each as {@code "left right"}
     * @param padToken the token that fills prompts up to {@code length}
     * @param length the number of ids every encoded prompt has (77)
     */
    public ClipTokenizer(Map<String, Integer> vocab, List<String> merges, String padToken, int length) {
        this.vocab = Map.copyOf(vocab);
        this.ranks = new HashMap<>();
        for (int i = 0; i < merges.size(); i++) {
            ranks.putIfAbsent(merges.get(i), i);
        }
        this.bos = id("<|startoftext|>");
        this.eos = id("<|endoftext|>");
        Integer p = vocab.get(padToken);
        this.pad = p == null ? eos : p;
        this.padText = p == null || padToken.startsWith("<|") ? null : padToken;
        this.length = length;
    }

    private int id(String token) {
        Integer id = vocab.get(token);
        if (id == null) {
            throw new IllegalArgumentException("the CLIP vocabulary has no " + token);
        }
        return id;
    }

    /**
     * Loads a tokenizer folder.
     *
     * @param dir a folder with {@code vocab.json} and {@code merges.txt}, or {@code tokenizer.json}
     * @param length the number of ids per prompt
     * @return the tokenizer
     * @throws IOException if the files cannot be read
     */
    public static ClipTokenizer load(Path dir, int length) throws IOException {
        return load(dir, length, null);
    }

    /**
     * Loads a tokenizer folder with a given pad token.
     *
     * @param dir a folder with {@code vocab.json} and {@code merges.txt}, or {@code tokenizer.json}
     * @param length the number of ids per prompt
     * @param pad the pad token, or {@code null} for the one the folder names (else the end token)
     * @return the tokenizer
     * @throws IOException if the files cannot be read
     */
    public static ClipTokenizer load(Path dir, int length, String pad) throws IOException {
        String padToken = "<|endoftext|>";
        for (String f : new String[] {"special_tokens_map.json", "tokenizer_config.json"}) {
            Path p = dir.resolve(f);
            if (Files.isRegularFile(p)) {
                Object v = Json.object(Json.read(p)).get("pad_token");
                if (v instanceof String s) {
                    padToken = s;
                } else if (v instanceof Map<?, ?> m && m.get("content") instanceof String s) {
                    padToken = s;
                }
            }
        }
        if (pad != null) {
            padToken = pad;
        }
        Map<String, Integer> vocab = new HashMap<>();
        List<String> merges = new ArrayList<>();
        Path vocabFile = dir.resolve("vocab.json");
        Path mergesFile = dir.resolve("merges.txt");
        if (Files.isRegularFile(vocabFile) && Files.isRegularFile(mergesFile)) {
            Json.object(Json.read(vocabFile)).forEach((k, v) -> vocab.put(k, ((Number) v).intValue()));
            for (String line : Files.readAllLines(mergesFile, StandardCharsets.UTF_8)) {
                if (!line.startsWith("#version") && !line.isBlank()) {
                    merges.add(line.strip());
                }
            }
        } else if (Files.isRegularFile(dir.resolve("tokenizer.json"))) {
            Map<String, Object> model = Json.object(Json.object(Json.read(dir.resolve("tokenizer.json"))).get("model"));
            Json.object(model.get("vocab")).forEach((k, v) -> vocab.put(k, ((Number) v).intValue()));
            for (Object m : Json.array(model.get("merges"))) {
                merges.add(m instanceof List<?> pair ? pair.get(0) + " " + pair.get(1) : (String) m);
            }
            Object added = Json.object(Json.read(dir.resolve("tokenizer.json"))).get("added_tokens");
            if (added != null) {
                for (Object a : Json.array(added)) {
                    Map<String, Object> t = Json.object(a);
                    vocab.put((String) t.get("content"), ((Number) t.get("id")).intValue());
                }
            }
        } else {
            throw new IOException("no CLIP tokenizer in " + dir + " (needs vocab.json and merges.txt, or tokenizer.json)");
        }
        return new ClipTokenizer(vocab, merges, padToken, length);
    }

    /**
     * Encodes a prompt as exactly {@code length} ids: start token, the prompt's tokens (cut
     * to fit), end token, then padding.
     *
     * @param text the prompt
     * @return the ids
     */
    public int[] encode(String text) {
        List<Integer> tokens = tokens(text);
        int[] ids = new int[length];
        java.util.Arrays.fill(ids, pad);
        ids[0] = bos;
        int n = Math.min(tokens.size(), length - 2);
        for (int i = 0; i < n; i++) {
            ids[i + 1] = tokens.get(i);
        }
        ids[n + 1] = eos;
        return ids;
    }

    /**
     * Returns how many tokens a prompt has, without the start and end tokens; prompts longer
     * than {@code length - 2} are cut.
     *
     * @param text the prompt
     * @return the token count
     */
    public int count(String text) {
        return tokens(text).size();
    }

    /**
     * Returns the end-of-text id.
     *
     * @return the id
     */
    public int eos() {
        return eos;
    }

    private List<Integer> tokens(String text) {
        String clean = unescape(text).replaceAll("\\s+", " ").strip().toLowerCase(Locale.ROOT);
        List<Integer> out = new ArrayList<>();
        if (padText == null) {
            words(clean, out);
            return out;
        }
        // a pad token such as SDXL's "!" is a special token: it is split out of the text first
        int from = 0;
        for (int at = clean.indexOf(padText); at >= 0; at = clean.indexOf(padText, from)) {
            words(clean.substring(from, at), out);
            out.add(pad);
            from = at + padText.length();
        }
        words(clean.substring(from), out);
        return out;
    }

    private void words(String clean, List<Integer> out) {
        Matcher m = WORDS.matcher(clean);
        while (m.find()) {
            String word = m.group();
            Integer special = word.startsWith("<|") ? vocab.get(word) : null;
            if (special != null) {
                out.add(special);
                continue;
            }
            StringBuilder sb = new StringBuilder();
            for (byte b : word.getBytes(StandardCharsets.UTF_8)) {
                sb.append(BYTE_CHARS[b & 0xff]);
            }
            for (int id : bpe(sb.toString())) {
                out.add(id);
            }
        }
    }

    private int[] bpe(String word) {
        int[] cached = cache.get(word);
        if (cached != null) {
            return cached;
        }
        List<String> parts = new ArrayList<>();
        int[] cps = word.codePoints().toArray();
        for (int i = 0; i < cps.length; i++) {
            String s = new String(cps, i, 1);
            parts.add(i == cps.length - 1 ? s + "</w>" : s);
        }
        while (parts.size() > 1) {
            int best = -1;
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i + 1 < parts.size(); i++) {
                Integer r = ranks.get(parts.get(i) + " " + parts.get(i + 1));
                if (r != null && r < bestRank) {
                    bestRank = r;
                    best = i;
                }
            }
            if (best < 0) {
                break;
            }
            String a = parts.get(best);
            String b = parts.get(best + 1);
            List<String> merged = new ArrayList<>(parts.size());
            for (int i = 0; i < parts.size(); i++) {
                if (i + 1 < parts.size() && parts.get(i).equals(a) && parts.get(i + 1).equals(b)) {
                    merged.add(a + b);
                    i++;
                } else {
                    merged.add(parts.get(i));
                }
            }
            parts = merged;
        }
        int[] ids = new int[parts.size()];
        for (int i = 0; i < ids.length; i++) {
            Integer id = vocab.get(parts.get(i));
            ids[i] = id == null ? vocab.getOrDefault("<|endoftext|>", 0) : id;
        }
        if (cache.size() < 10_000) {
            cache.put(word, ids);
        }
        return ids;
    }

    private static String unescape(String s) {
        return s.replace("&quot;", "\"").replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&amp;", "&");
    }

    /** GPT-2's reversible map from bytes to printable characters. */
    private static String[] byteChars() {
        String[] out = new String[256];
        boolean[] direct = new boolean[256];
        for (int b = '!'; b <= '~'; b++) {
            direct[b] = true;
        }
        for (int b = 0xA1; b <= 0xAC; b++) {
            direct[b] = true;
        }
        for (int b = 0xAE; b <= 0xFF; b++) {
            direct[b] = true;
        }
        int next = 256;
        for (int b = 0; b < 256; b++) {
            out[b] = direct[b] ? Character.toString(b) : Character.toString(next++);
        }
        return out;
    }
}
