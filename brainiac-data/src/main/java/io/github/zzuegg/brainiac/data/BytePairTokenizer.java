package io.github.zzuegg.brainiac.data;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Byte-pair encoding: splits text into tokens that are often whole words or common word
 * pieces, the way GPT models read text. It is not a neural network but a fixed table,
 * learned once by counting:
 *
 * <ol>
 * <li>Every text starts as its UTF-8 bytes, tokens 0 to 255, so any text can be encoded.
 * <li>Training finds the pair of neighbouring tokens that occurs most often, say
 *     {@code "t" + "h"}, gives it a new id ({@code "th"}), and repeats until the vocabulary
 *     has the size asked for. Later merges build on earlier ones: {@code " th" + "e"}.
 * <li>Encoding replays the merges in the order they were learned.
 * </ol>
 *
 * <p>Text is first cut into words, numbers, punctuation and spaces (a word keeps the space
 * in front of it), and pairs never cross those cuts, so tokens stay meaningful pieces.
 *
 * <pre>{@code
 * BytePairTokenizer bpe = BytePairTokenizer.train(text, 1024);   // or Tokenizer.train("bpe:1024", text)
 * int[] ids = bpe.encode("To be, or not to be");   // about 7 tokens instead of 19 characters
 * String back = bpe.decode(ids);                    // the same text
 * }</pre>
 */
public final class BytePairTokenizer implements Tokenizer {

    private final int[][] merges;
    private final byte[][] bytes;
    private final Map<Long, Integer> rank;
    private final Map<String, int[]> cache = new ConcurrentHashMap<>();

    private BytePairTokenizer(int[][] merges) {
        this.merges = merges;
        this.bytes = new byte[256 + merges.length][];
        this.rank = new HashMap<>(merges.length * 2);
        for (int b = 0; b < 256; b++) {
            bytes[b] = new byte[] {(byte) b};
        }
        for (int i = 0; i < merges.length; i++) {
            int a = merges[i][0];
            int b = merges[i][1];
            if (a < 0 || b < 0 || a >= 256 + i || b >= 256 + i) {
                throw new IllegalArgumentException("merge " + i + " uses token " + Math.max(a, b)
                        + " before it exists");
            }
            byte[] joined = new byte[bytes[a].length + bytes[b].length];
            System.arraycopy(bytes[a], 0, joined, 0, bytes[a].length);
            System.arraycopy(bytes[b], 0, joined, bytes[a].length, bytes[b].length);
            bytes[256 + i] = joined;
            rank.put(pair(a, b), i);
        }
    }

    private static long pair(int a, int b) {
        return ((long) a << 32) | (b & 0xffffffffL);
    }

    /**
     * Learns merges from a text.
     *
     * @param text the training text
     * @param vocabularySize the number of tokens wanted, at least 256 (the bytes); training
     *     stops early when no pair occurs twice
     * @return the tokenizer
     */
    public static BytePairTokenizer train(String text, int vocabularySize) {
        return new BytePairTokenizer(learn(text, vocabularySize, false));
    }

    /**
     * Learns merges. By count, the most frequent pair is merged (BPE); by likelihood, the
     * pair with the highest {@code count(ab) / (count(a) count(b))} (WordPiece), which
     * prefers pairs whose parts rarely occur apart.
     */
    static int[][] learn(String text, int vocabularySize, boolean likelihood) {
        if (vocabularySize < 256) {
            throw new IllegalArgumentException("the vocabulary needs at least the 256 bytes, got " + vocabularySize);
        }
        Map<String, Integer> counts = Pretokenizer.counts(text);
        int n = counts.size();
        int[][] words = new int[n][];
        int[] freq = new int[n];
        long[] tokens = new long[vocabularySize];
        int w = 0;
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            byte[] b = Pretokenizer.utf8(e.getKey());
            words[w] = new int[b.length];
            for (int i = 0; i < b.length; i++) {
                words[w][i] = b[i] & 0xff;
                tokens[words[w][i]] += e.getValue();
            }
            freq[w++] = e.getValue();
        }
        Map<Long, Long> pairCounts = new HashMap<>();
        Map<Long, Set<Integer>> where = new HashMap<>();
        for (int i = 0; i < n; i++) {
            count(words[i], freq[i], i, pairCounts, where);
        }
        // by likelihood, pairs of rare pieces score highest; a floor on the count, lowered only
        // when no pair reaches it, keeps the vocabulary for pieces common enough to matter
        long total = 0;
        for (int i = 0; i < n; i++) {
            total += (long) freq[i] * words[i].length;
        }
        long floor = likelihood ? Math.max(2, total / (4L * vocabularySize)) : 2;
        List<int[]> merges = new ArrayList<>();
        while (256 + merges.size() < vocabularySize) {
            long best = 0;
            double bestScore = 0;
            for (Map.Entry<Long, Long> e : pairCounts.entrySet()) {
                long c = e.getValue();
                if (c < floor) {
                    continue;
                }
                long p = e.getKey();
                double score = likelihood
                        ? c / ((double) tokens[(int) (p >>> 32)] * tokens[(int) p]) : c;
                if (score > bestScore || (score == bestScore && p < best)) {
                    best = p;
                    bestScore = score;
                }
            }
            if (bestScore == 0) {
                if (floor > 2) {
                    floor = Math.max(2, floor / 2);
                    continue;
                }
                break;
            }
            int a = (int) (best >>> 32);
            int b = (int) best;
            int id = 256 + merges.size();
            merges.add(new int[] {a, b});
            Set<Integer> affected = where.remove(best);
            for (int i : affected) {
                for (int t : words[i]) {
                    tokens[t] -= freq[i];
                }
                count(words[i], -freq[i], -1, pairCounts, where);
                words[i] = merge(words[i], a, b, id);
                count(words[i], freq[i], i, pairCounts, where);
                for (int t : words[i]) {
                    tokens[t] += freq[i];
                }
            }
            pairCounts.remove(best);
        }
        return merges.toArray(new int[0][]);
    }

    private static void count(int[] word, int freq, int index, Map<Long, Long> counts, Map<Long, Set<Integer>> where) {
        for (int i = 0; i + 1 < word.length; i++) {
            long p = pair(word[i], word[i + 1]);
            long c = counts.merge(p, (long) freq, Long::sum);
            if (c == 0) {
                counts.remove(p);
            }
            if (index >= 0) {
                where.computeIfAbsent(p, k -> new HashSet<>()).add(index);
            }
        }
    }

    private static int[] merge(int[] word, int a, int b, int id) {
        int[] out = new int[word.length];
        int n = 0;
        for (int i = 0; i < word.length; i++) {
            if (i + 1 < word.length && word[i] == a && word[i + 1] == b) {
                out[n++] = id;
                i++;
            } else {
                out[n++] = word[i];
            }
        }
        return n == out.length ? out : java.util.Arrays.copyOf(out, n);
    }

    /**
     * Splits a text into token ids.
     *
     * @param text the text
     * @return the ids, each in {@code [0, size())}
     */
    @Override
    public int[] encode(String text) {
        List<int[]> parts = new ArrayList<>();
        int total = 0;
        for (String chunk : Pretokenizer.chunks(text)) {
            int[] ids = cache.computeIfAbsent(chunk, this::encodeChunk);
            parts.add(ids);
            total += ids.length;
        }
        int[] out = new int[total];
        int at = 0;
        for (int[] p : parts) {
            System.arraycopy(p, 0, out, at, p.length);
            at += p.length;
        }
        return out;
    }

    private int[] encodeChunk(String chunk) {
        byte[] b = chunk.getBytes(StandardCharsets.UTF_8);
        int[] word = new int[b.length];
        for (int i = 0; i < b.length; i++) {
            word[i] = b[i] & 0xff;
        }
        while (word.length > 1) {
            int best = Integer.MAX_VALUE;
            int at = -1;
            for (int i = 0; i + 1 < word.length; i++) {
                Integer r = rank.get(pair(word[i], word[i + 1]));
                if (r != null && r < best) {
                    best = r;
                    at = i;
                }
            }
            if (at < 0) {
                break;
            }
            word = merge(word, merges[best][0], merges[best][1], 256 + best);
        }
        return word;
    }

    /**
     * Turns token ids back into text.
     *
     * @param ids the ids
     * @return the text; bytes that do not form valid UTF-8 become replacement characters
     */
    @Override
    public String decode(int... ids) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(ids.length * 4);
        for (int id : ids) {
            out.writeBytes(tokenBytes(id));
        }
        return out.toString(StandardCharsets.UTF_8);
    }

    /**
     * The bytes a token stands for.
     *
     * @param id the token id
     * @return a copy of its bytes
     */
    public byte[] tokenBytes(int id) {
        if (id < 0 || id >= bytes.length) {
            throw new IllegalArgumentException("token " + id + " is outside [0, " + bytes.length + ")");
        }
        return bytes[id].clone();
    }

    /** @return the number of tokens: 256 bytes plus one per merge */
    @Override
    public int size() {
        return bytes.length;
    }

    /** @return the merges in the order they were learned, each a pair of token ids */
    public int[][] merges() {
        int[][] copy = new int[merges.length][];
        for (int i = 0; i < merges.length; i++) {
            copy[i] = merges[i].clone();
        }
        return copy;
    }

    /**
     * Builds a tokenizer from merges, for example read back from a file.
     *
     * @param merges pairs of token ids; merge {@code i} creates token {@code 256 + i}
     * @return the tokenizer
     */
    public static BytePairTokenizer of(int[][] merges) {
        int[][] copy = new int[merges.length][];
        for (int i = 0; i < merges.length; i++) {
            if (merges[i].length != 2) {
                throw new IllegalArgumentException("merge " + i + " needs two ids, got " + merges[i].length);
            }
            copy[i] = merges[i].clone();
        }
        return new BytePairTokenizer(copy);
    }

    @Override
    public String kind() {
        return "bpe";
    }

    @Override
    public void save(Path path) throws IOException {
        List<String> lines = new ArrayList<>(merges.length);
        for (int[] m : merges) {
            lines.add(m[0] + " " + m[1]);
        }
        Pretokenizer.write(path, kind(), lines);
    }

    static BytePairTokenizer read(List<String> lines) {
        int[][] merges = new int[lines.size()][];
        for (int i = 0; i < merges.length; i++) {
            String[] p = lines.get(i).trim().split(" ");
            merges[i] = new int[] {Integer.parseInt(p[0]), Integer.parseInt(p[1])};
        }
        return new BytePairTokenizer(merges);
    }

    /** The byte strings of all tokens, in id order, as a vocabulary of pieces. */
    static List<byte[]> pieces(int[][] merges) {
        return new BytePairTokenizer(merges).pieces();
    }

    private List<byte[]> pieces() {
        List<byte[]> out = new ArrayList<>(merges.length);
        for (int i = 256; i < bytes.length; i++) {
            out.add(bytes[i]);
        }
        return out;
    }

    @Override
    public String toString() {
        return "BytePairTokenizer(" + size() + " tokens)";
    }
}
