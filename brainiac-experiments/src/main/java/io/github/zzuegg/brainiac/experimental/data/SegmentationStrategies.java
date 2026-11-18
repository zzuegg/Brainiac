package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.BytePairTokenizer;
import io.github.zzuegg.brainiac.data.Tokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Inference-time segmentation over a fixed vocabulary: PathPiece (Schmidt et al., EMNLP 2024,
 * <a href="https://arxiv.org/abs/2402.18376">arXiv:2402.18376</a>) and the comparison of
 * inference methods by Uzan et al. (2024, <a href="https://arxiv.org/abs/2403.01289">arXiv:2403.01289</a>).
 * A vocabulary allows many segmentations of a word, and which one a tokenizer picks matters
 * about as much as the vocabulary itself. This wrapper turns any vocabulary (a list of token
 * strings) into a {@link Tokenizer} with one of four strategies:
 * <ul>
 * <li>{@link Strategy#MERGE_ORDER}: replay BPE merges by rank (BPE's own rule; needs merges);</li>
 * <li>{@link Strategy#LONGEST_PREFIX}: greedy longest match from the left (WordPiece's rule);</li>
 * <li>{@link Strategy#LEAST_TOKENS}: the fewest tokens, a shortest path over the word's
 *     character lattice (PathPiece; ties to the longer first token);</li>
 * <li>{@link Strategy#MAX_LIKELIHOOD}: the most likely segmentation under unigram token
 *     log-probabilities (Viterbi, the Unigram/SentencePiece rule).</li>
 * </ul>
 * Text is pre-tokenized into words that keep their leading space, as for the core BPE, and
 * segmented within words. Every character of the input must be a token.
 */
public final class SegmentationStrategies implements Tokenizer {

    /** The segmentation rule. */
    public enum Strategy {
        /** BPE merges by rank. */
        MERGE_ORDER,
        /** Greedy longest prefix. */
        LONGEST_PREFIX,
        /** Fewest tokens (PathPiece). */
        LEAST_TOKENS,
        /** Highest unigram likelihood. */
        MAX_LIKELIHOOD
    }

    private final String[] pieces;
    private final Map<String, Integer> index = new HashMap<>();
    private final Strategy strategy;
    private final int[][] merges;
    private final double[] logProbs;
    private final int maxLength;
    private final Map<Long, Integer> rank = new HashMap<>();

    private SegmentationStrategies(String[] pieces, Strategy strategy, int[][] merges, double[] logProbs) {
        this.pieces = pieces.clone();
        int longest = 1;
        for (int i = 0; i < pieces.length; i++) {
            if (pieces[i] != null && !pieces[i].isEmpty()) {
                index.putIfAbsent(pieces[i], i);
                longest = Math.max(longest, pieces[i].length());
            }
        }
        this.maxLength = longest;
        this.strategy = Objects.requireNonNull(strategy);
        if (strategy == Strategy.MERGE_ORDER && merges == null) {
            throw new IllegalArgumentException("MERGE_ORDER needs merges");
        }
        this.merges = merges;
        this.logProbs = logProbs;
        for (int i = 0; merges != null && i < merges.length; i++) {
            rank.put(((long) merges[i][0] << 32) | merges[i][1], i);
        }
    }

    /**
     * A vocabulary of token strings.
     *
     * @param pieces each token's text by id ({@code null} for unusable ids)
     * @param strategy the strategy (not {@code MERGE_ORDER}); {@code MAX_LIKELIHOOD} starts
     *     with uniform log-probabilities, see {@link #withUnigramCounts}
     * @return the tokenizer
     */
    public static SegmentationStrategies of(String[] pieces, Strategy strategy) {
        return new SegmentationStrategies(pieces, strategy, null, uniform(pieces.length));
    }

    /**
     * The vocabulary and merges of a core BPE tokenizer (ASCII tokens only).
     *
     * @param bpe the tokenizer
     * @param strategy the strategy
     * @return the tokenizer
     */
    public static SegmentationStrategies of(BytePairTokenizer bpe, Strategy strategy) {
        String[] pieces = new String[bpe.size()];
        for (int t = 0; t < pieces.length; t++) {
            byte[] b = bpe.tokenBytes(t);
            boolean ascii = true;
            for (byte x : b) {
                ascii &= x >= 0;
            }
            pieces[t] = ascii ? new String(b, StandardCharsets.US_ASCII) : null;
        }
        return new SegmentationStrategies(pieces, strategy, bpe.merges(), uniform(pieces.length));
    }

    private static double[] uniform(int n) {
        double[] lp = new double[n];
        Arrays.fill(lp, -Math.log(n));
        return lp;
    }

    /**
     * Estimates unigram log-probabilities by segmenting a text with the fewest tokens and
     * counting (add-one smoothing), for {@code MAX_LIKELIHOOD}.
     *
     * @param text the text
     * @return a copy with those log-probabilities
     */
    public SegmentationStrategies withUnigramCounts(String text) {
        SegmentationStrategies counter = new SegmentationStrategies(pieces, Strategy.LEAST_TOKENS, merges, logProbs);
        double[] counts = new double[pieces.length];
        Arrays.fill(counts, 1);
        for (int id : counter.encode(text)) {
            counts[id]++;
        }
        double total = Arrays.stream(counts).sum();
        double[] lp = new double[counts.length];
        for (int i = 0; i < lp.length; i++) {
            lp[i] = Math.log(counts[i] / total);
        }
        return new SegmentationStrategies(pieces, strategy, merges, lp);
    }

    /**
     * @param s another strategy
     * @return the same vocabulary with that strategy
     */
    public SegmentationStrategies withStrategy(Strategy s) {
        return new SegmentationStrategies(pieces, s, merges, logProbs);
    }

    /** @return the strategy */
    public Strategy strategy() {
        return strategy;
    }

    @Override
    public int[] encode(String text) {
        List<Integer> out = new ArrayList<>();
        for (String w : LengthMaxTokenizer.words(text)) {
            for (int id : segment(w)) {
                out.add(id);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Segments one word.
     *
     * @param w the word
     * @return its token ids
     */
    public int[] segment(String w) {
        return switch (strategy) {
            case MERGE_ORDER -> mergeOrder(w);
            case LONGEST_PREFIX -> longestPrefix(w);
            case LEAST_TOKENS -> lattice(w, false);
            case MAX_LIKELIHOOD -> lattice(w, true);
        };
    }

    private int id(String s) {
        Integer id = index.get(s);
        if (id == null) {
            throw new IllegalArgumentException("'" + s + "' is not in the vocabulary");
        }
        return id;
    }

    private int[] longestPrefix(String w) {
        List<Integer> out = new ArrayList<>();
        int i = 0;
        while (i < w.length()) {
            int len = Math.min(maxLength, w.length() - i);
            Integer id = null;
            for (; len > 0; len--) {
                id = index.get(w.substring(i, i + len));
                if (id != null) {
                    break;
                }
            }
            if (id == null) {
                throw new IllegalArgumentException("'" + w.charAt(i) + "' is not in the vocabulary");
            }
            out.add(id);
            i += len;
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Shortest path (fewest tokens, ties to longer last-step tokens) or Viterbi. */
    private int[] lattice(String w, boolean likelihood) {
        int n = w.length();
        double[] best = new double[n + 1];
        int[] from = new int[n + 1];
        int[] tok = new int[n + 1];
        Arrays.fill(best, Double.POSITIVE_INFINITY);
        best[0] = 0;
        for (int end = 1; end <= n; end++) {
            for (int len = 1; len <= Math.min(maxLength, end); len++) {
                Integer id = index.get(w.substring(end - len, end));
                if (id == null || best[end - len] == Double.POSITIVE_INFINITY) {
                    continue;
                }
                double cost = best[end - len] + (likelihood ? -logProbs[id] : 1 - 1e-6 * len);
                if (cost < best[end]) {
                    best[end] = cost;
                    from[end] = end - len;
                    tok[end] = id;
                }
            }
            if (best[end] == Double.POSITIVE_INFINITY) {
                throw new IllegalArgumentException("'" + w.substring(0, end) + "' cannot be segmented");
            }
        }
        List<Integer> out = new ArrayList<>();
        for (int at = n; at > 0; at = from[at]) {
            out.add(0, tok[at]);
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private int[] mergeOrder(String w) {
        int base = pieces.length - merges.length;
        List<Integer> s = new ArrayList<>();
        for (char c : w.toCharArray()) {
            s.add(id(String.valueOf(c)));
        }
        while (s.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int at = -1;
            for (int i = 0; i + 1 < s.size(); i++) {
                Integer r = rank.get(((long) s.get(i) << 32) | s.get(i + 1));
                if (r != null && r < bestRank) {
                    bestRank = r;
                    at = i;
                }
            }
            if (at < 0) {
                break;
            }
            s.set(at, base + bestRank);
            s.remove(at + 1);
        }
        return s.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public String decode(int... ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            sb.append(pieces[id]);
        }
        return sb.toString();
    }

    @Override
    public int size() {
        return pieces.length;
    }

    @Override
    public String kind() {
        return "segmentation";
    }

    /**
     * Writes one token per line ({@code null} tokens as empty lines).
     *
     * @param path the file
     * @throws IOException if it cannot be written
     */
    @Override
    public void save(Path path) throws IOException {
        List<String> lines = new ArrayList<>();
        for (String p : pieces) {
            lines.add(p == null ? "" : p.replace("\\", "\\\\").replace("\n", "\\n"));
        }
        Files.write(path, lines, StandardCharsets.UTF_8);
    }

    @Override
    public String toString() {
        return "SegmentationStrategies(" + strategy + ", " + pieces.length + " tokens)";
    }
}
