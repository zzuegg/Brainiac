package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.Tokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The Length-MAX tokenizer (Dong et al. 2025, <a href="https://arxiv.org/abs/2511.20849">arXiv:2511.20849</a>).
 * BPE picks merges by pair frequency; Length-MAX picks its vocabulary to minimise the average
 * number of tokens per character, which the paper casts as a graph partitioning problem and
 * solves greedily: starting from the characters, every step scores each candidate string by
 * {@code frequency x length} (ties to the more frequent, then the longer), adds the best one and
 * substitutes it in place in the segmented corpus, until the vocabulary has the size asked for.
 * Candidates are concatenations of neighbouring current tokens up to {@code maxLength}
 * characters that occur at least twice, inside the pre-tokenized words (a word keeps the space
 * in front of it, as for the core BPE, so tokens never cross words). Encoding is the paper's
 * left-most longest match.
 *
 * <p>Token ids: the characters of the training text first (sorted), then the learned strings in
 * the order they were added.
 */
public final class LengthMaxTokenizer implements Tokenizer {

    private static final Pattern WORDS = Pattern.compile(" ?[^ ]+| +");

    private final List<String> pieces;
    private final Map<String, Integer> index = new HashMap<>();
    private final int maxLength;

    private LengthMaxTokenizer(List<String> pieces) {
        this.pieces = List.copyOf(pieces);
        int longest = 1;
        for (int i = 0; i < pieces.size(); i++) {
            index.put(pieces.get(i), i);
            longest = Math.max(longest, pieces.get(i).length());
        }
        this.maxLength = longest;
    }

    /**
     * Splits a text into the pre-tokenized words the tokenizers work within.
     *
     * @param text the text
     * @return the words, each with its leading space
     */
    public static List<String> words(String text) {
        List<String> out = new ArrayList<>();
        Matcher m = WORDS.matcher(text);
        while (m.find()) {
            out.add(m.group());
        }
        return out;
    }

    /**
     * Learns a vocabulary.
     *
     * @param text the training text
     * @param learned the strings to add beyond the characters
     * @param maxLength the longest token, in characters
     * @return the tokenizer
     */
    public static LengthMaxTokenizer train(String text, int learned, int maxLength) {
        TreeSet<String> chars = new TreeSet<>();
        for (char c : text.toCharArray()) {
            chars.add(String.valueOf(c));
        }
        List<String> vocab = new ArrayList<>(chars);
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (String w : words(text)) {
            counts.merge(w, 1, Integer::sum);
        }
        List<List<String>> segs = new ArrayList<>();
        List<Integer> freq = new ArrayList<>();
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            List<String> s = new ArrayList<>();
            for (char c : e.getKey().toCharArray()) {
                s.add(String.valueOf(c));
            }
            segs.add(s);
            freq.add(e.getValue());
        }
        java.util.Set<String> known = new java.util.HashSet<>(vocab);
        for (int step = 0; step < learned; step++) {
            Map<String, Long> score = new HashMap<>();
            for (int w = 0; w < segs.size(); w++) {
                List<String> s = segs.get(w);
                for (int i = 0; i < s.size(); i++) {
                    StringBuilder sb = new StringBuilder(s.get(i));
                    for (int j = i + 1; j < s.size(); j++) {
                        sb.append(s.get(j));
                        if (sb.length() > maxLength) {
                            break;
                        }
                        score.merge(sb.toString(), (long) freq.get(w), Long::sum);
                    }
                }
            }
            String best = null;
            long bestScore = -1;
            long bestFreq = -1;
            for (Map.Entry<String, Long> e : score.entrySet()) {
                long f = e.getValue();
                if (f < 2 || known.contains(e.getKey())) {
                    continue;
                }
                long sc = f * e.getKey().length();
                if (sc > bestScore || sc == bestScore && (f > bestFreq || f == bestFreq
                        && (e.getKey().length() > best.length() || e.getKey().length() == best.length()
                        && e.getKey().compareTo(best) < 0))) {
                    best = e.getKey();
                    bestScore = sc;
                    bestFreq = f;
                }
            }
            if (best == null) {
                break;
            }
            vocab.add(best);
            known.add(best);
            for (int w = 0; w < segs.size(); w++) {
                segs.set(w, substitute(segs.get(w), best));
            }
        }
        return new LengthMaxTokenizer(vocab);
    }

    /** Replaces, left to right, every run of tokens that spells {@code t} by {@code t}. */
    private static List<String> substitute(List<String> s, String t) {
        List<String> out = new ArrayList<>();
        int i = 0;
        while (i < s.size()) {
            StringBuilder sb = new StringBuilder();
            int j = i;
            while (j < s.size() && sb.length() < t.length()) {
                sb.append(s.get(j++));
            }
            if (sb.toString().equals(t)) {
                out.add(t);
                i = j;
            } else {
                out.add(s.get(i++));
            }
        }
        return out;
    }

    /** @return every token's text, by id */
    public List<String> pieces() {
        return pieces;
    }

    @Override
    public int[] encode(String text) {
        List<Integer> out = new ArrayList<>();
        int i = 0;
        while (i < text.length()) {
            int end = Math.min(text.length(), i + maxLength);
            // tokens never cross a word start (a space after the first character)
            int stop = text.indexOf(' ', i + 1);
            if (stop > 0) {
                end = Math.min(end, stop);
            }
            Integer id = null;
            int len = end - i;
            for (; len > 0; len--) {
                id = index.get(text.substring(i, i + len));
                if (id != null) {
                    break;
                }
            }
            if (id == null) {
                throw new IllegalArgumentException("character '" + text.charAt(i) + "' is not in the vocabulary");
            }
            out.add(id);
            i += len;
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public String decode(int... ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            sb.append(pieces.get(id));
        }
        return sb.toString();
    }

    @Override
    public int size() {
        return pieces.size();
    }

    @Override
    public String kind() {
        return "lengthmax";
    }

    /**
     * Writes one token per line, escaped as Java string content ({@code \n}, {@code \\}).
     *
     * @param path the file
     * @throws IOException if it cannot be written
     */
    @Override
    public void save(Path path) throws IOException {
        List<String> lines = new ArrayList<>();
        for (String p : pieces) {
            lines.add(p.replace("\\", "\\\\").replace("\n", "\\n"));
        }
        Files.write(path, lines, StandardCharsets.UTF_8);
    }

    /**
     * Reads a tokenizer written by {@link #save}.
     *
     * @param path the file
     * @return the tokenizer
     * @throws IOException if it cannot be read
     */
    public static LengthMaxTokenizer load(Path path) throws IOException {
        List<String> pieces = new ArrayList<>();
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < line.length(); i++) {
                char c = line.charAt(i);
                if (c == '\\' && i + 1 < line.length()) {
                    char n = line.charAt(++i);
                    sb.append(n == 'n' ? '\n' : n);
                } else {
                    sb.append(c);
                }
            }
            pieces.add(sb.toString());
        }
        return new LengthMaxTokenizer(pieces);
    }

    @Override
    public String toString() {
        return "LengthMaxTokenizer(" + pieces.size() + " tokens)";
    }
}
