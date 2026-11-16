package io.github.zzuegg.brainiac.data;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The unigram language model tokenizer of SentencePiece (T5, many multilingual models).
 * Every piece has a probability, and a text is split into the pieces whose product of
 * probabilities is highest. Training starts from many candidate pieces (frequent
 * substrings), fits the probabilities by expectation-maximization over all possible splits,
 * and repeatedly drops the pieces whose loss would hurt the likelihood of the text least,
 * until the vocabulary has the size asked for. The 256 bytes always stay, so any text can
 * be split.
 */
public final class UnigramTokenizer implements Tokenizer {

    /** The longest candidate piece, in bytes. */
    private static final int MAX_PIECE = 16;
    private static final double FLOOR = -30;

    private final Pieces pieces;
    private final double[] logp;
    private final Map<String, int[]> cache = new ConcurrentHashMap<>();

    private UnigramTokenizer(Pieces pieces, double[] logp) {
        if (logp.length != pieces.bytes.length) {
            throw new IllegalArgumentException("need one probability per piece");
        }
        this.pieces = pieces;
        this.logp = logp;
    }

    /**
     * Learns pieces and their probabilities from a text.
     *
     * @param text the training text
     * @param vocabularySize the number of tokens wanted, at least 256
     * @return the tokenizer
     */
    public static UnigramTokenizer train(String text, int vocabularySize) {
        if (vocabularySize < 256) {
            throw new IllegalArgumentException("the vocabulary needs at least the 256 bytes, got " + vocabularySize);
        }
        Map<String, Integer> counts = Pretokenizer.counts(text);
        byte[][] chunks = new byte[counts.size()][];
        int[] freq = new int[chunks.length];
        int c = 0;
        Map<String, Long> candidates = new HashMap<>();
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            byte[] b = Pretokenizer.utf8(e.getKey());
            chunks[c] = b;
            freq[c++] = e.getValue();
            for (int s = 0; s < b.length; s++) {
                for (int end = s + 2; end <= Math.min(b.length, s + MAX_PIECE); end++) {
                    candidates.merge(Pretokenizer.key(b, s, end), (long) e.getValue(), Long::sum);
                }
            }
        }
        List<Map.Entry<String, Long>> ranked = new ArrayList<>();
        for (Map.Entry<String, Long> e : candidates.entrySet()) {
            if (e.getValue() >= 2) {
                ranked.add(e);
            }
        }
        ranked.sort(Comparator.<Map.Entry<String, Long>>comparingLong(e -> -e.getValue() * e.getKey().length())
                .thenComparing(Map.Entry::getKey));
        int seed = Math.min(ranked.size(), Math.max(8 * (vocabularySize - 256), 4096));
        List<byte[]> longer = new ArrayList<>(seed);
        for (int i = 0; i < seed; i++) {
            longer.add(Pretokenizer.unkey(ranked.get(i).getKey()));
        }
        Pieces vocab = new Pieces(longer);
        double[] lp = new double[vocab.bytes.length];
        Arrays.fill(lp, Math.log(1.0 / lp.length));
        while (true) {
            for (int iteration = 0; iteration < 2; iteration++) {
                lp = maximize(expected(vocab, lp, chunks, freq));
            }
            int target = vocabularySize - 256;
            int current = vocab.bytes.length - 256;
            if (current <= target) {
                break;
            }
            int keep = Math.max(target, (int) (current * 0.8));
            double[] loss = removalLoss(vocab, lp, chunks, freq);
            Integer[] order = new Integer[current];
            for (int i = 0; i < current; i++) {
                order[i] = 256 + i;
            }
            double[] score = loss;
            Arrays.sort(order, Comparator.<Integer>comparingDouble(i -> -score[i]).thenComparing(i -> i));
            Integer[] kept = Arrays.copyOf(order, keep);
            Arrays.sort(kept);
            List<byte[]> next = new ArrayList<>(keep);
            double[] nextLp = new double[256 + keep];
            System.arraycopy(lp, 0, nextLp, 0, 256);
            for (int i = 0; i < keep; i++) {
                next.add(vocab.bytes[kept[i]]);
                nextLp[256 + i] = lp[kept[i]];
            }
            vocab = new Pieces(next);
            lp = nextLp;
        }
        // most likely pieces first, for readable ids
        int n = vocab.bytes.length - 256;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = 256 + i;
        }
        double[] fin = lp;
        Arrays.sort(order, Comparator.<Integer>comparingDouble(i -> -fin[i]).thenComparing(i -> i));
        List<byte[]> sorted = new ArrayList<>(n);
        double[] sortedLp = new double[256 + n];
        System.arraycopy(lp, 0, sortedLp, 0, 256);
        for (int i = 0; i < n; i++) {
            sorted.add(vocab.bytes[order[i]]);
            sortedLp[256 + i] = lp[order[i]];
        }
        return new UnigramTokenizer(new Pieces(sorted), sortedLp);
    }

    /** Expected uses of each piece over all splits of all chunks (forward-backward). */
    private static double[] expected(Pieces vocab, double[] lp, byte[][] chunks, int[] freq) {
        double[] counts = new double[lp.length];
        for (int w = 0; w < chunks.length; w++) {
            byte[] b = chunks[w];
            int n = b.length;
            double[] alpha = new double[n + 1];
            double[] beta = new double[n + 1];
            Arrays.fill(alpha, Double.NEGATIVE_INFINITY);
            Arrays.fill(beta, Double.NEGATIVE_INFINITY);
            alpha[0] = 0;
            int[][] ids = new int[n][];
            for (int s = 0; s < n; s++) {
                int max = Math.min(n, s + vocab.longest);
                ids[s] = new int[max - s];
                for (int e = s + 1; e <= max; e++) {
                    ids[s][e - s - 1] = vocab.find(b, s, e);
                }
            }
            for (int s = 0; s < n; s++) {
                if (alpha[s] == Double.NEGATIVE_INFINITY) {
                    continue;
                }
                for (int k = 0; k < ids[s].length; k++) {
                    int id = ids[s][k];
                    if (id >= 0) {
                        alpha[s + k + 1] = logAdd(alpha[s + k + 1], alpha[s] + lp[id]);
                    }
                }
            }
            beta[n] = 0;
            for (int s = n - 1; s >= 0; s--) {
                for (int k = 0; k < ids[s].length; k++) {
                    int id = ids[s][k];
                    if (id >= 0) {
                        beta[s] = logAdd(beta[s], lp[id] + beta[s + k + 1]);
                    }
                }
            }
            double z = alpha[n];
            for (int s = 0; s < n; s++) {
                for (int k = 0; k < ids[s].length; k++) {
                    int id = ids[s][k];
                    if (id >= 0) {
                        counts[id] += freq[w] * Math.exp(alpha[s] + lp[id] + beta[s + k + 1] - z);
                    }
                }
            }
        }
        return counts;
    }

    private static double[] maximize(double[] counts) {
        double total = 0;
        for (double c : counts) {
            total += c;
        }
        double[] lp = new double[counts.length];
        for (int i = 0; i < lp.length; i++) {
            lp[i] = counts[i] > 0 ? Math.max(Math.log(counts[i] / total), FLOOR) : FLOOR;
        }
        return lp;
    }

    /**
     * How much the text's log-likelihood would drop without each piece: its uses in the best
     * splits times the gap between it and the best split of its own bytes without it.
     */
    private static double[] removalLoss(Pieces vocab, double[] lp, byte[][] chunks, int[] freq) {
        double[] uses = new double[lp.length];
        for (int w = 0; w < chunks.length; w++) {
            for (int id : viterbi(vocab, lp, chunks[w], -1)) {
                uses[id] += freq[w];
            }
        }
        double[] loss = new double[lp.length];
        for (int id = 256; id < lp.length; id++) {
            if (uses[id] == 0) {
                continue;
            }
            int[] alt = viterbi(vocab, lp, vocab.bytes[id], id);
            double altLp = 0;
            for (int a : alt) {
                altLp += lp[a];
            }
            loss[id] = uses[id] * (lp[id] - altLp);
        }
        return loss;
    }

    /** The most likely split of some bytes, optionally without one piece. */
    private static int[] viterbi(Pieces vocab, double[] lp, byte[] b, int without) {
        int n = b.length;
        double[] best = new double[n + 1];
        int[] from = new int[n + 1];
        int[] piece = new int[n + 1];
        Arrays.fill(best, Double.NEGATIVE_INFINITY);
        best[0] = 0;
        for (int s = 0; s < n; s++) {
            if (best[s] == Double.NEGATIVE_INFINITY) {
                continue;
            }
            for (int e = s + 1; e <= Math.min(n, s + vocab.longest); e++) {
                int id = vocab.find(b, s, e);
                if (id >= 0 && id != without && best[s] + lp[id] > best[e]) {
                    best[e] = best[s] + lp[id];
                    from[e] = s;
                    piece[e] = id;
                }
            }
        }
        int count = 0;
        for (int at = n; at > 0; at = from[at]) {
            count++;
        }
        int[] out = new int[count];
        for (int at = n; at > 0; at = from[at]) {
            out[--count] = piece[at];
        }
        return out;
    }

    private static double logAdd(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) {
            return b;
        }
        if (b == Double.NEGATIVE_INFINITY) {
            return a;
        }
        double max = Math.max(a, b);
        return max + Math.log1p(Math.exp(-Math.abs(a - b)));
    }

    @Override
    public int[] encode(String text) {
        List<int[]> parts = new ArrayList<>();
        int total = 0;
        for (String chunk : Pretokenizer.chunks(text)) {
            int[] ids = cache.computeIfAbsent(chunk, k -> viterbi(pieces, logp, Pretokenizer.utf8(k), -1));
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

    @Override
    public String decode(int... ids) {
        return pieces.decode(ids);
    }

    @Override
    public int size() {
        return pieces.bytes.length;
    }

    /**
     * @param id a token id
     * @return the natural log of the token's probability
     */
    public double logProbability(int id) {
        return logp[id];
    }

    @Override
    public String kind() {
        return "unigram";
    }

    @Override
    public void save(Path path) throws IOException {
        List<String> lines = new ArrayList<>(logp.length);
        for (int i = 0; i < logp.length; i++) {
            lines.add(Pretokenizer.hex(pieces.bytes[i]) + " " + logp[i]);
        }
        Pretokenizer.write(path, kind(), lines);
    }

    static UnigramTokenizer read(List<String> lines) {
        List<byte[]> longer = new ArrayList<>();
        double[] lp = new double[lines.size()];
        for (int i = 0; i < lines.size(); i++) {
            String[] p = lines.get(i).trim().split(" ");
            if (i >= 256) {
                longer.add(Pretokenizer.unhex(p[0]));
            }
            lp[i] = Double.parseDouble(p[1]);
        }
        return new UnigramTokenizer(new Pieces(longer), lp);
    }

    @Override
    public String toString() {
        return "UnigramTokenizer(" + size() + " tokens)";
    }
}
