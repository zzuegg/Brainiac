package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The character tokenizer of VITS and MMS-TTS ({@code VitsTokenizer}): lower-cased text,
 * characters outside the vocabulary dropped, each character one token, with a blank
 * (token 0) between them.
 */
public final class VitsTokenizer {

    private final Map<String, Integer> vocab;
    /** Vocabulary entries longer than one character, longest first, matched before lower-casing. */
    private final List<String> multi;
    private final boolean addBlank;
    private final boolean normalize;
    private final boolean phonemize;
    private final boolean uroman;
    private final String language;

    /**
     * Creates a tokenizer.
     *
     * @param vocab characters to ids
     * @param addBlank whether to put token 0 between characters
     * @param normalize whether to lower-case and drop unknown characters
     * @param phonemize whether the model expects espeak phonemes (not supported)
     * @param uroman whether the model expects romanized text
     * @param language the model's language code, or {@code null}
     */
    public VitsTokenizer(Map<String, Integer> vocab, boolean addBlank, boolean normalize, boolean phonemize,
            boolean uroman, String language) {
        this.vocab = new LinkedHashMap<>(vocab);
        this.multi = new ArrayList<>();
        for (String k : vocab.keySet()) {
            if (k.codePointCount(0, k.length()) > 1) {
                multi.add(k);
            }
        }
        this.addBlank = addBlank;
        this.normalize = normalize;
        this.phonemize = phonemize;
        this.uroman = uroman;
        this.language = language;
    }

    /**
     * Loads {@code vocab.json} and {@code tokenizer_config.json} of a Hugging Face folder.
     *
     * @param dir the folder
     * @return the tokenizer
     */
    public static VitsTokenizer load(Path dir) {
        Path v = dir.resolve("vocab.json");
        if (!Files.isRegularFile(v)) {
            throw new IllegalArgumentException(dir + " has no vocab.json; a VITS folder needs its tokenizer files");
        }
        try {
            Map<String, Integer> vocab = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : Json.object(Json.read(v)).entrySet()) {
                vocab.put(e.getKey(), ((Number) e.getValue()).intValue());
            }
            Path tc = dir.resolve("tokenizer_config.json");
            Map<String, Object> c = Files.isRegularFile(tc) ? Json.object(Json.read(tc)) : Map.of();
            return new VitsTokenizer(vocab, flag(c, "add_blank", true), flag(c, "normalize", true),
                    flag(c, "phonemize", false), flag(c, "is_uroman", false),
                    c.get("language") instanceof String s ? s : null);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read the tokenizer of " + dir + ": " + e.getMessage(), e);
        }
    }

    private static boolean flag(Map<String, Object> c, String key, boolean otherwise) {
        return c.get(key) instanceof Boolean b ? b : otherwise;
    }

    /**
     * The model's language as the tokenizer names it (MMS uses ISO 639-3, such as {@code eng}).
     *
     * @return the code, or {@code null}
     */
    public String language() {
        return language;
    }

    /**
     * Turns text into the model's input.
     *
     * @param text the text
     * @return token ids; empty if no character of the text is in the vocabulary
     * @throws IllegalArgumentException if the model needs phonemes or romanized text that
     *     Brainiac cannot make
     */
    public int[] encode(String text) {
        if (phonemize) {
            throw new IllegalArgumentException("this VITS model reads espeak phonemes, which Brainiac cannot make;"
                    + " use an MMS-TTS model (facebook/mms-tts-*), which reads characters");
        }
        String s = normalize ? normalized(text) : text;
        if (uroman && s.codePoints().anyMatch(cp -> cp > 0x024F && Character.isLetter(cp))) {
            throw new IllegalArgumentException("this model reads romanized text (uroman); write the text in Latin"
                    + " letters");
        }
        List<Integer> ids = new ArrayList<>();
        int i = 0;
        while (i < s.length()) {
            String hit = null;
            for (String m : multi) {
                if (s.startsWith(m, i)) {
                    hit = m;
                    break;
                }
            }
            String ch = hit != null ? hit : s.substring(i, i + Character.charCount(s.codePointAt(i)));
            Integer id = vocab.get(ch);
            if (id != null) {
                ids.add(id);
            } else if (!normalize) {
                Integer unk = vocab.get("<unk>");
                if (unk != null) {
                    ids.add(unk);
                }
            }
            i += ch.length();
        }
        int n = ids.size();
        if (n == 0) {
            return new int[0];
        }
        if (!addBlank) {
            return ids.stream().mapToInt(Integer::intValue).toArray();
        }
        int[] out = new int[2 * n + 1];
        for (int k = 0; k < n; k++) {
            out[2 * k + 1] = ids.get(k);
        }
        return out;
    }

    /** Lower-cased except where a vocabulary entry matches, then only vocabulary characters, stripped. */
    private String normalized(String text) {
        if (language != null && language.equals("ron")) {
            text = text.replace('ț', 'ţ');
        }
        StringBuilder lower = new StringBuilder();
        int i = 0;
        while (i < text.length()) {
            String hit = null;
            for (String k : vocab.keySet()) {
                if (!k.isEmpty() && text.startsWith(k, i)) {
                    hit = k;
                    break;
                }
            }
            if (hit != null) {
                lower.append(hit);
                i += hit.length();
            } else {
                int cp = text.codePointAt(i);
                lower.append(new String(Character.toChars(cp)).toLowerCase(Locale.ROOT));
                i += Character.charCount(cp);
            }
        }
        StringBuilder kept = new StringBuilder();
        lower.codePoints().forEach(cp -> {
            String c = new String(Character.toChars(cp));
            if (vocab.containsKey(c)) {
                kept.append(c);
            }
        });
        return kept.toString().strip();
    }
}
