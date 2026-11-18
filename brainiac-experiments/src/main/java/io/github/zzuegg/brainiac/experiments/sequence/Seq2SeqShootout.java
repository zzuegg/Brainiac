package io.github.zzuegg.brainiac.experiments.sequence;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.PointerNetwork;
import io.github.zzuegg.brainiac.experimental.nn.Seq2SeqAttention;
import io.github.zzuegg.brainiac.experimental.nn.TransformerSeq2Seq;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 6 items 2.7 and 2.8: recurrent sequence-to-sequence models with and without attention,
 * pointer networks and pointer-generator copying, against a two-layer transformer
 * encoder-decoder ({@link TransformerSeq2Seq}), on toy transduction tasks scored by exact
 * match, including lengths longer than any seen in training:
 *
 * <ul>
 *   <li><b>reversal</b> of symbol strings, trained on lengths 4 to 8;</li>
 *   <li><b>date normalisation</b>: {@code dd/mm/yy}, {@code MON d, yy} or {@code yy-mm-dd} to
 *       {@code yymmdd};</li>
 *   <li><b>sorting</b> numbers, trained on 5 to 8 of them ({@link PointerNetwork} against the
 *       attentional seq2seq and the transformer emitting the sorted tokens);</li>
 *   <li><b>convex hull</b> of random points, trained on 6 to 8 points, as hull indices in
 *       counter-clockwise order from the leftmost point, then stop;</li>
 *   <li><b>copy-heavy summaries</b> of {@code CountingTasks}-style sequences over 40 content
 *       tokens, a quarter of which never occur in training: emit the stride, the first and
 *       the last token. The {@link PointerNetwork.Generator pointer-generator} cannot generate
 *       the unseen tokens and must copy them; the baseline generates from all 50 ids.</li>
 * </ul>
 */
public final class Seq2SeqShootout {

    /** The compared models. */
    public enum Architecture {
        /** GRU seq2seq with a fixed context */
        SEQ2SEQ,
        /** GRU seq2seq with Bahdanau attention */
        BAHDANAU,
        /** GRU seq2seq with Luong attention */
        LUONG,
        /** two-layer transformer encoder-decoder */
        TRANSFORMER
    }

    /** A token-to-token model behind one interface. */
    private interface Transducer {
        Var loss(int[][] source, int[][] target);

        int[][] greedy(int[][] source, int length);

        List<Var> parameters();
    }

    private Seq2SeqShootout() {
    }

    private static Transducer create(Architecture m, int sourceVocab, int targetVocab, long seed) {
        if (m == Architecture.TRANSFORMER) {
            TransformerSeq2Seq t = TransformerSeq2Seq.of(sourceVocab, targetVocab, 32, 4, 2, seed);
            return new Transducer() {
                @Override
                public Var loss(int[][] source, int[][] target) {
                    return t.loss(source, target);
                }

                @Override
                public int[][] greedy(int[][] source, int length) {
                    return t.greedy(source, length);
                }

                @Override
                public List<Var> parameters() {
                    return t.parameters();
                }
            };
        }
        Seq2SeqAttention.Attention a = switch (m) {
            case BAHDANAU -> Seq2SeqAttention.Attention.BAHDANAU;
            case LUONG -> Seq2SeqAttention.Attention.LUONG;
            default -> Seq2SeqAttention.Attention.NONE;
        };
        Seq2SeqAttention s = Seq2SeqAttention.of(sourceVocab, targetVocab, 16, 48, a, seed);
        return new Transducer() {
            @Override
            public Var loss(int[][] source, int[][] target) {
                return s.loss(s.embedSource(source), target);
            }

            @Override
            public int[][] greedy(int[][] source, int length) {
                return s.greedy(s.embedSource(source), length);
            }

            @Override
            public List<Var> parameters() {
                return s.parameters();
            }
        };
    }

    /** A task: makes {source, target} pairs of a given length. */
    private interface Task {
        int[][][] batch(int n, int length, SplittableRandom r);
    }

    private static void train(Transducer t, Task task, int minLength, int maxLength, int steps, long seed) {
        Optimizer opt = Optimizer.adam(5e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int len = minLength + r.nextInt(maxLength - minLength + 1);
            int[][][] d = task.batch(32, len, r);
            t.loss(d[0], d[1]).backward();
            opt.step(t.parameters());
        }
    }

    /** {@code {exact match, share of positions right}}. */
    private static double[] matches(int[][] predicted, int[][] target) {
        int right = 0;
        int total = 0;
        for (int i = 0; i < target.length; i++) {
            for (int t = 0; t < target[i].length; t++) {
                right += predicted[i][t] == target[i][t] ? 1 : 0;
                total++;
            }
        }
        return new double[] {exactMatch(predicted, target), (double) right / total};
    }

    private static double exactMatch(int[][] predicted, int[][] target) {
        int hit = 0;
        for (int i = 0; i < target.length; i++) {
            hit += Arrays.equals(predicted[i], target[i]) ? 1 : 0;
        }
        return (double) hit / target.length;
    }

    // ------------------------------------------------------------------ reversal

    /** Symbols of the reversal task. */
    public static final int SYMBOLS = 10;

    private static int[][][] reversal(int n, int length, SplittableRandom r) {
        int[][] src = new int[n][length];
        int[][] tgt = new int[n][length];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < length; t++) {
                src[i][t] = r.nextInt(SYMBOLS);
            }
            for (int t = 0; t < length; t++) {
                tgt[i][t] = src[i][length - 1 - t];
            }
        }
        return new int[][][] {src, tgt};
    }

    /**
     * Trains on reversal of 4 to 8 symbols and scores exact match by length.
     *
     * @param m the model
     * @param steps Adam steps of 32 strings
     * @param seed the seed
     * @param lengths the test lengths
     * @return exact-match accuracy per test length (256 strings each)
     */
    public static double[] reversal(Architecture m, int steps, long seed, int... lengths) {
        Transducer t = create(m, SYMBOLS, SYMBOLS, seed);
        train(t, Seq2SeqShootout::reversal, 4, 8, steps, seed);
        double[] acc = new double[lengths.length];
        for (int k = 0; k < lengths.length; k++) {
            int[][][] d = reversal(256, lengths[k], new SplittableRandom(seed + 100 + k));
            acc[k] = exactMatch(t.greedy(d[0], lengths[k]), d[1]);
        }
        return acc;
    }

    // ------------------------------------------------------------------ dates

    private static final int MONTH = 10;
    private static final int SLASH = 22;
    private static final int DASH = 23;
    private static final int COMMA = 24;
    private static final int PAD = 25;
    private static final int DATE_LENGTH = 8;

    private static int[][][] dates(int n, int ignored, SplittableRandom r) {
        int[][] src = new int[n][DATE_LENGTH];
        int[][] tgt = new int[n][6];
        for (int i = 0; i < n; i++) {
            int d = 1 + r.nextInt(28);
            int m = 1 + r.nextInt(12);
            int y = r.nextInt(100);
            List<Integer> s = new ArrayList<>();
            switch (r.nextInt(3)) {
                case 0 -> {
                    s.addAll(List.of(d / 10, d % 10, SLASH, m / 10, m % 10, SLASH, y / 10, y % 10));
                }
                case 1 -> {
                    s.add(MONTH + m - 1);
                    if (d >= 10) {
                        s.add(d / 10);
                    }
                    s.addAll(List.of(d % 10, COMMA, y / 10, y % 10));
                }
                default -> s.addAll(List.of(y / 10, y % 10, DASH, m / 10, m % 10, DASH, d / 10, d % 10));
            }
            while (s.size() < DATE_LENGTH) {
                s.add(PAD);
            }
            for (int t = 0; t < DATE_LENGTH; t++) {
                src[i][t] = s.get(t);
            }
            tgt[i] = new int[] {y / 10, y % 10, m / 10, m % 10, d / 10, d % 10};
        }
        return new int[][][] {src, tgt};
    }

    /**
     * Trains on date normalisation.
     *
     * @param m the model
     * @param steps Adam steps of 32 dates
     * @param seed the seed
     * @return exact-match accuracy on 512 fresh dates
     */
    public static double dates(Architecture m, int steps, long seed) {
        Transducer t = create(m, PAD + 1, 10, seed);
        train(t, Seq2SeqShootout::dates, DATE_LENGTH, DATE_LENGTH, steps, seed);
        int[][][] d = dates(512, 0, new SplittableRandom(seed + 100));
        return exactMatch(t.greedy(d[0], 6), d[1]);
    }

    // ------------------------------------------------------------------ sorting

    /** The value range of the sorting task. */
    public static final int VALUES = 50;

    private static int[][][] sorting(int n, int length, SplittableRandom r) {
        int[][] src = new int[n][length];
        int[][] order = new int[n][length];
        int[][] sorted = new int[n][length];
        for (int i = 0; i < n; i++) {
            List<Integer> pool = new ArrayList<>();
            for (int v = 0; v < VALUES; v++) {
                pool.add(v);
            }
            java.util.Collections.shuffle(pool, new java.util.Random(r.nextLong()));
            Integer[] idx = new Integer[length];
            for (int t = 0; t < length; t++) {
                src[i][t] = pool.get(t);
                idx[t] = t;
            }
            final int[] row = src[i];
            Arrays.sort(idx, (a, b) -> Integer.compare(row[a], row[b]));
            for (int t = 0; t < length; t++) {
                order[i][t] = idx[t];
                sorted[i][t] = row[idx[t]];
            }
        }
        return new int[][][] {src, sorted, order};
    }

    private static Var values(int[][] src) {
        int n = src.length;
        int len = src[0].length;
        float[] f = new float[n * len * VALUES];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < len; t++) {
                f[(i * len + t) * VALUES + src[i][t]] = 1f;
            }
        }
        return Var.of(Tensor.of(Shape.of(n, len, VALUES), f));
    }

    /**
     * Sorting with a pointer network, each number given as a one-hot vector: trained on 5 to 8
     * numbers.
     *
     * @param steps Adam steps of 32 lists
     * @param seed the seed
     * @param lengths the test lengths
     * @return {@code [length][exact match, position accuracy]} of the sorted output
     */
    public static double[][] sortWithPointer(int steps, long seed, int... lengths) {
        PointerNetwork p = PointerNetwork.of(VALUES, 48, false, seed);
        Optimizer opt = Optimizer.adam(5e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[][][] d = sorting(32, 5 + r.nextInt(4), r);
            p.loss(values(d[0]), d[2]).backward();
            opt.step(p.parameters());
        }
        double[][] acc = new double[lengths.length][];
        for (int k = 0; k < lengths.length; k++) {
            int[][][] d = sorting(256, lengths[k], new SplittableRandom(seed + 100 + k));
            int[][] pos = p.decode(values(d[0]), lengths[k], true);
            int[][] out = new int[pos.length][lengths[k]];
            for (int i = 0; i < pos.length; i++) {
                for (int t = 0; t < lengths[k]; t++) {
                    out[i][t] = d[0][i][pos[i][t]];
                }
            }
            acc[k] = matches(out, d[1]);
        }
        return acc;
    }

    /**
     * Sorting with a token model emitting the sorted values: trained on 5 to 8 numbers.
     *
     * @param m the model
     * @param steps Adam steps of 32 lists
     * @param seed the seed
     * @param lengths the test lengths
     * @return {@code [length][exact match, position accuracy]}
     */
    public static double[][] sortWithTokens(Architecture m, int steps, long seed, int... lengths) {
        Transducer t = create(m, VALUES, VALUES, seed);
        train(t, Seq2SeqShootout::sorting, 5, 8, steps, seed);
        double[][] acc = new double[lengths.length][];
        for (int k = 0; k < lengths.length; k++) {
            int[][][] d = sorting(256, lengths[k], new SplittableRandom(seed + 100 + k));
            acc[k] = matches(t.greedy(d[0], lengths[k]), d[1]);
        }
        return acc;
    }

    // ------------------------------------------------------------------ convex hull

    /** Hull indices counter-clockwise from the leftmost (then lowest) point: Andrew's monotone chain. */
    static int[] hull(float[][] pts) {
        Integer[] idx = new Integer[pts.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> pts[a][0] != pts[b][0] ? Float.compare(pts[a][0], pts[b][0])
                : Float.compare(pts[a][1], pts[b][1]));
        int[] h = new int[2 * pts.length];
        int k = 0;
        for (int i = 0; i < idx.length; i++) {
            while (k >= 2 && cross(pts[h[k - 2]], pts[h[k - 1]], pts[idx[i]]) <= 0) {
                k--;
            }
            h[k++] = idx[i];
        }
        for (int i = idx.length - 2, lower = k + 1; i >= 0; i--) {
            while (k >= lower && cross(pts[h[k - 2]], pts[h[k - 1]], pts[idx[i]]) <= 0) {
                k--;
            }
            h[k++] = idx[i];
        }
        return Arrays.copyOf(h, k - 1);
    }

    private static double cross(float[] o, float[] a, float[] b) {
        return (double) (a[0] - o[0]) * (b[1] - o[1]) - (double) (a[1] - o[1]) * (b[0] - o[0]);
    }

    /** {points [n, len, 2], pointer targets (stop = len), token targets (stop = maxPoints)}, padded with -1. */
    private static Object[] hullBatch(int n, int len, int maxPoints, SplittableRandom r) {
        float[] f = new float[n * len * 2];
        int width = len + 1;
        int[][] pointer = new int[n][width];
        int[][] tokens = new int[n][width];
        for (int i = 0; i < n; i++) {
            float[][] pts = new float[len][2];
            for (int j = 0; j < len; j++) {
                pts[j][0] = r.nextFloat();
                pts[j][1] = r.nextFloat();
                f[(i * len + j) * 2] = pts[j][0];
                f[(i * len + j) * 2 + 1] = pts[j][1];
            }
            int[] h = hull(pts);
            Arrays.fill(pointer[i], -1);
            Arrays.fill(tokens[i], -1);
            for (int t = 0; t < h.length; t++) {
                pointer[i][t] = h[t];
                tokens[i][t] = h[t];
            }
            pointer[i][h.length] = len;
            tokens[i][h.length] = maxPoints;
        }
        return new Object[] {Var.of(Tensor.of(Shape.of(n, len, 2), f)), pointer, tokens};
    }

    /** Exact match up to and including the stop; entries after it are ignored. */
    private static double hullMatch(int[][] predicted, int[][] target, int stop) {
        int hit = 0;
        for (int i = 0; i < target.length; i++) {
            boolean ok = true;
            for (int t = 0; t < target[i].length && target[i][t] >= 0; t++) {
                ok &= predicted[i][t] == target[i][t];
                if (target[i][t] == stop) {
                    break;
                }
            }
            hit += ok ? 1 : 0;
        }
        return (double) hit / target.length;
    }

    /**
     * Convex hull with a pointer network (with a stop key) or with an attentional seq2seq that
     * emits point indices as tokens (indices 0 to 9 plus a stop token).
     *
     * @param pointer whether to use the pointer network
     * @param steps Adam steps of 32 point sets
     * @param seed the seed
     * @param lengths the test point counts
     * @return exact-match accuracy of the hull per point count
     */
    public static double[] convexHull(boolean pointer, int steps, long seed, int... lengths) {
        int maxPoints = 10;
        PointerNetwork p = pointer ? PointerNetwork.of(2, 48, true, seed) : null;
        Seq2SeqAttention s = pointer ? null : Seq2SeqAttention.continuous(2, maxPoints + 1, 16, 48,
                Seq2SeqAttention.Attention.BAHDANAU, seed);
        List<Var> params = pointer ? p.parameters() : s.parameters();
        Optimizer opt = Optimizer.adam(5e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int st = 0; st < steps; st++) {
            int len = 6 + r.nextInt(3);
            Object[] d = hullBatch(32, len, maxPoints, r);
            Var loss = pointer ? p.loss((Var) d[0], (int[][]) d[1]) : s.loss((Var) d[0], (int[][]) d[2]);
            loss.backward();
            opt.step(params);
        }
        double[] acc = new double[lengths.length];
        for (int k = 0; k < lengths.length; k++) {
            int len = lengths[k];
            Object[] d = hullBatch(256, len, maxPoints, new SplittableRandom(seed + 100 + k));
            acc[k] = pointer ? hullMatch(p.decode((Var) d[0], len + 1, false), (int[][]) d[1], len)
                    : hullMatch(s.greedy((Var) d[0], len + 1), (int[][]) d[2], maxPoints);
        }
        return acc;
    }

    // ------------------------------------------------------------------ copy-heavy summaries

    /** Control tokens (strides 1 to 5 are ids 1 to 5); content tokens are ids 10 to 49. */
    public static final int CONTROL = 10;
    /** The whole id space. */
    public static final int ALL_TOKENS = 50;
    /** Content ids below this appear in training; the rest only at test time. */
    public static final int SEEN = 40;

    private static int[][][] summaries(int n, int length, int contentTokens, SplittableRandom r) {
        int[][] src = new int[n][length];
        int[][] tgt = new int[n][3];
        for (int i = 0; i < n; i++) {
            int stride = 1 + r.nextInt(5);
            int v = r.nextInt(contentTokens);
            for (int t = 0; t < length; t++) {
                src[i][t] = CONTROL + v;
                v = (v + stride) % contentTokens;
            }
            tgt[i] = new int[] {stride, src[i][0], src[i][length - 1]};
        }
        return new int[][][] {src, tgt};
    }

    /**
     * Copy-heavy summaries: pointer-generator (generator vocabulary of the 10 control tokens and
     * the 30 content tokens seen in training; the other 10 can only be copied) against the
     * Bahdanau seq2seq generating from all 50 ids. Training uses length 8 and content ids 10 to
     * 39; tests use all 40 content ids, so a quarter of the tokens were never seen.
     *
     * @param pointerGenerator whether to use the pointer-generator
     * @param steps Adam steps of 32 sequences
     * @param seed the seed
     * @param lengths the test lengths
     * @return {@code [length][exact match, content-token accuracy]}
     */
    public static double[][] summaries(boolean pointerGenerator, int steps, long seed, int... lengths) {
        PointerNetwork.Generator g = pointerGenerator ? new PointerNetwork.Generator(ALL_TOKENS, SEEN, 16, 48, seed)
                : null;
        Transducer base = pointerGenerator ? null : create(Architecture.BAHDANAU, ALL_TOKENS, ALL_TOKENS, seed);
        List<Var> params = pointerGenerator ? g.parameters() : base.parameters();
        Optimizer opt = Optimizer.adam(5e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[][][] d = summaries(32, 8, SEEN - CONTROL, r);
            (pointerGenerator ? g.loss(d[0], d[1]) : base.loss(d[0], d[1])).backward();
            opt.step(params);
        }
        double[][] out = new double[lengths.length][];
        for (int k = 0; k < lengths.length; k++) {
            int[][][] d = summaries(256, lengths[k], ALL_TOKENS - CONTROL,
                    new SplittableRandom(seed + 100 + k));
            int[][] pred = pointerGenerator ? g.greedy(d[0], 3) : base.greedy(d[0], 3);
            int content = 0;
            for (int i = 0; i < pred.length; i++) {
                content += (pred[i][1] == d[1][i][1] ? 1 : 0) + (pred[i][2] == d[1][i][2] ? 1 : 0);
            }
            out[k] = new double[] {exactMatch(pred, d[1]), content / (2.0 * pred.length)};
        }
        return out;
    }
}
