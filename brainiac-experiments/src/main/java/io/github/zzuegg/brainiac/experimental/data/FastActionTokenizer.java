package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.Tokenizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * FAST, frequency-space action sequence tokenization (Pertsch et al.,
 * <a href="https://arxiv.org/abs/2501.09747">2501.09747</a>): turns a robot action chunk
 * {@code [H][D]} into a short sequence of discrete tokens for an autoregressive policy.
 *
 * <ol>
 *   <li>A discrete cosine transform (orthonormal DCT-II) over time, per action dimension.
 *       Smooth trajectories put almost all their energy in the first few coefficients.</li>
 *   <li>Scale by {@code gamma} and round to integers: most high-frequency coefficients become
 *       0. The coefficients are flattened lowest frequency first, all dimensions of one
 *       frequency together, so the zeros collect at the end.</li>
 *   <li>Byte-pair encoding over the integer symbols, trained on chunks from the data set,
 *       merges the frequent patterns (above all runs of zeros) into single tokens.</li>
 * </ol>
 *
 * <p>Compared with the usual per-step uniform binning (one token per action dimension per
 * step), FAST needs a fraction of the tokens per chunk, and each token carries more
 * information, which makes autoregressive training and decoding of high-frequency
 * control both cheaper and easier. Decoding is the exact inverse up to the rounding.
 *
 * <p>It is a core {@link Tokenizer} over a text form of the rounded coefficients (one
 * character per coefficient, see {@link #coefficientText}), and adds the chunk-level
 * {@link #encodeChunk} and {@link #decodeChunk}.
 */
public final class FastActionTokenizer implements Tokenizer, ActionTokenizer {
    private static final char BASE = 'Ā';
    private final int horizon;
    private final int actionSize;
    private final float gamma;
    private final int range;
    private final List<int[]> merges;               // {left, right, new}
    private final Map<Long, Integer> mergeRank = new HashMap<>();
    private final List<int[]> expansion = new ArrayList<>();   // token -> symbols

    private FastActionTokenizer(int horizon, int actionSize, float gamma, int range, List<int[]> merges) {
        this.horizon = horizon;
        this.actionSize = actionSize;
        this.gamma = gamma;
        this.range = range;
        this.merges = List.copyOf(merges);
        for (int s = 0; s < 2 * range + 1; s++) {
            expansion.add(new int[] {s});
        }
        for (int i = 0; i < this.merges.size(); i++) {
            int[] m = this.merges.get(i);
            mergeRank.put(key(m[0], m[1]), i);
            int[] a = expansion.get(m[0]);
            int[] b = expansion.get(m[1]);
            int[] ab = java.util.Arrays.copyOf(a, a.length + b.length);
            System.arraycopy(b, 0, ab, a.length, b.length);
            expansion.add(ab);
        }
    }

    /**
     * Trains the tokenizer's merges on example chunks.
     *
     * @param chunks example chunks, each {@code [horizon][actionSize]}, values about in [-1, 1]
     * @param gamma the scale before rounding; larger keeps more detail and more tokens
     * @param vocabulary the target vocabulary size, at least {@code 2 * range + 1}
     * @return the tokenizer
     */
    public static FastActionTokenizer train(List<float[][]> chunks, float gamma, int vocabulary) {
        int horizon = chunks.getFirst().length;
        int actionSize = chunks.getFirst()[0].length;
        int range = Math.max(1, (int) Math.ceil(gamma * Math.sqrt(horizon)));
        FastActionTokenizer base = new FastActionTokenizer(horizon, actionSize, gamma, range, List.of());
        List<List<Integer>> corpus = new ArrayList<>();
        for (float[][] c : chunks) {
            List<Integer> seq = new ArrayList<>();
            for (int s : base.symbols(c)) {
                seq.add(s);
            }
            corpus.add(seq);
        }
        List<int[]> merges = new ArrayList<>();
        int next = 2 * range + 1;
        while (next < vocabulary) {
            Map<Long, Integer> counts = new HashMap<>();
            for (List<Integer> seq : corpus) {
                for (int i = 0; i + 1 < seq.size(); i++) {
                    counts.merge(key(seq.get(i), seq.get(i + 1)), 1, Integer::sum);
                }
            }
            long best = -1;
            int bestCount = 1;
            for (Map.Entry<Long, Integer> e : counts.entrySet()) {
                if (e.getValue() > bestCount || (e.getValue() == bestCount && best >= 0 && e.getKey() < best)) {
                    best = e.getKey();
                    bestCount = e.getValue();
                }
            }
            if (best < 0) {
                break;   // nothing occurs twice any more
            }
            int left = (int) (best >>> 32);
            int right = (int) best;
            for (List<Integer> seq : corpus) {
                merge(seq, left, right, next);
            }
            merges.add(new int[] {left, right, next});
            next++;
        }
        return new FastActionTokenizer(horizon, actionSize, gamma, range, merges);
    }

    private static long key(int a, int b) {
        return ((long) a << 32) | (b & 0xffffffffL);
    }

    private static void merge(List<Integer> seq, int left, int right, int token) {
        for (int i = 0; i + 1 < seq.size(); i++) {
            if (seq.get(i) == left && seq.get(i + 1) == right) {
                seq.set(i, token);
                seq.remove(i + 1);
            }
        }
    }

    /**
     * The orthonormal DCT-II of a sequence.
     *
     * @param x the values
     * @return the coefficients
     */
    public static double[] dct(double[] x) {
        int n = x.length;
        double[] out = new double[n];
        for (int k = 0; k < n; k++) {
            double s = 0;
            for (int i = 0; i < n; i++) {
                s += x[i] * Math.cos(Math.PI * (i + 0.5) * k / n);
            }
            out[k] = s * Math.sqrt((k == 0 ? 1.0 : 2.0) / n);
        }
        return out;
    }

    /**
     * The inverse of {@link #dct}.
     *
     * @param c the coefficients
     * @return the values
     */
    public static double[] idct(double[] c) {
        int n = c.length;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int k = 0; k < n; k++) {
                s += c[k] * Math.sqrt((k == 0 ? 1.0 : 2.0) / n) * Math.cos(Math.PI * (i + 0.5) * k / n);
            }
            out[i] = s;
        }
        return out;
    }

    /** The rounded, flattened DCT coefficients of a chunk as symbols {@code 0..2 range}. */
    private int[] symbols(float[][] chunk) {
        int[] out = new int[horizon * actionSize];
        for (int d = 0; d < actionSize; d++) {
            double[] x = new double[horizon];
            for (int t = 0; t < horizon; t++) {
                x[t] = chunk[t][d];
            }
            double[] c = dct(x);
            for (int k = 0; k < horizon; k++) {
                long q = Math.round(c[k] * gamma);
                out[k * actionSize + d] = (int) Math.max(-range, Math.min(range, q)) + range;
            }
        }
        return out;
    }

    /**
     * Tokenizes a chunk.
     *
     * @param chunk {@code [horizon][actionSize]}
     * @return the tokens
     */
    @Override
    public int[] encodeChunk(float[][] chunk) {
        if (chunk.length != horizon || chunk[0].length != actionSize) {
            throw new IllegalArgumentException("expected a " + horizon + "x" + actionSize + " chunk");
        }
        return applyMerges(symbols(chunk));
    }

    private int[] applyMerges(int[] symbols) {
        List<Integer> seq = new ArrayList<>();
        for (int s : symbols) {
            seq.add(s);
        }
        while (seq.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            for (int i = 0; i + 1 < seq.size(); i++) {
                Integer r = mergeRank.get(key(seq.get(i), seq.get(i + 1)));
                if (r != null && r < bestRank) {
                    bestRank = r;
                }
            }
            if (bestRank == Integer.MAX_VALUE) {
                break;
            }
            int[] m = merges.get(bestRank);
            merge(seq, m[0], m[1], m[2]);
        }
        return seq.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Turns tokens back into a chunk. Missing coefficients (a truncated sequence) are 0 and
     * extra ones are ignored, so any token sequence decodes.
     *
     * @param tokens the tokens
     * @return {@code [horizon][actionSize]}
     */
    @Override
    public float[][] decodeChunk(int[] tokens) {
        int[] q = new int[horizon * actionSize];
        int n = 0;
        for (int tok : tokens) {
            if (tok < 0 || tok >= size()) {
                continue;
            }
            for (int s : expansion.get(tok)) {
                if (n < q.length) {
                    q[n++] = s - range;
                }
            }
        }
        float[][] out = new float[horizon][actionSize];
        for (int d = 0; d < actionSize; d++) {
            double[] c = new double[horizon];
            for (int k = 0; k < horizon; k++) {
                c[k] = q[k * actionSize + d] / (double) gamma;
            }
            double[] x = idct(c);
            for (int t = 0; t < horizon; t++) {
                out[t][d] = (float) x[t];
            }
        }
        return out;
    }

    /**
     * The text form of a chunk that {@link #encode(String)} reads: one character per rounded
     * coefficient.
     *
     * @param chunk {@code [horizon][actionSize]}
     * @return the text
     */
    public String coefficientText(float[][] chunk) {
        StringBuilder sb = new StringBuilder();
        for (int s : symbols(chunk)) {
            sb.append((char) (BASE + s));
        }
        return sb.toString();
    }

    @Override
    public int[] encode(String text) {
        int[] s = new int[text.length()];
        for (int i = 0; i < s.length; i++) {
            int v = text.charAt(i) - BASE;
            if (v < 0 || v > 2 * range) {
                throw new IllegalArgumentException("not a coefficient character at " + i);
            }
            s[i] = v;
        }
        return applyMerges(s);
    }

    @Override
    public String decode(int... ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            for (int s : expansion.get(id)) {
                sb.append((char) (BASE + s));
            }
        }
        return sb.toString();
    }

    @Override
    public int size() {
        return expansion.size();
    }

    @Override
    public String kind() {
        return "fast";
    }

    @Override
    public void save(Path path) throws IOException {
        List<String> lines = new ArrayList<>();
        lines.add(horizon + " " + actionSize + " " + gamma + " " + range);
        for (int[] m : merges) {
            lines.add(m[0] + " " + m[1]);
        }
        Tokenizer.write(path, kind(), lines);
    }

    /**
     * Reads what {@link #save} wrote, as a core {@link Tokenizer.Reader}; register it with
     * {@code Tokenizer.register("fast", FastActionTokenizer::read)}.
     *
     * @param lines the file body
     * @return the tokenizer
     */
    public static FastActionTokenizer read(List<String> lines) {
        String[] h = lines.getFirst().trim().split(" ");
        int range = Integer.parseInt(h[3]);
        List<int[]> merges = new ArrayList<>();
        int next = 2 * range + 1;
        for (String l : lines.subList(1, lines.size())) {
            if (l.isBlank()) {
                continue;
            }
            String[] p = l.trim().split(" ");
            merges.add(new int[] {Integer.parseInt(p[0]), Integer.parseInt(p[1]), next++});
        }
        return new FastActionTokenizer(Integer.parseInt(h[0]), Integer.parseInt(h[1]), Float.parseFloat(h[2]), range,
                merges);
    }

    @Override
    public int horizon() {
        return horizon;
    }

    @Override
    public int actionSize() {
        return actionSize;
    }
}
