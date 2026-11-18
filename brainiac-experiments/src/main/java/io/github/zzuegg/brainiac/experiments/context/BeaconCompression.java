package io.github.zzuegg.brainiac.experiments.context;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.BeaconCompressor;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Learned context compression (8.4.10, {@link BeaconCompressor}) on key-value retrieval, the
 * {@link LongContextNeedle} task with a learned reader: a context of two-token slots (a key
 * {@code k} and its value {@code v}, or filler; four distinct keys planted) is encoded token by
 * token (each token's embedding plus its predecessor's, so a value carries its key), turned into
 * a memory, and a reader answers {@code ?k} by attending to the memory. Every arm trains the
 * same encoder and reader end to end on 32-token contexts and is tested on 32 and on 128 tokens
 * (4 times the training length). Memories:
 * <ul>
 * <li>full context (every token: the upper bound);</li>
 * <li>sliding window (the last 8 tokens);</li>
 * <li>mean pooling of groups of 4 or 8 tokens (the POOL mode of {@code CompressiveKvMemory},
 *     the Compressive Transformer's compression; trained end to end here, which that
 *     inference-only cache cannot be), 4x and 8x;</li>
 * <li>Activation Beacon: 2 or 1 learned beacons per 8-token chunk, 4x and 8x;</li>
 * <li>gist tokens: the whole context into 2 gists (16x at training length, 64x at 128).</li>
 * </ul>
 * Reported: accuracy and memory entries per question.
 */
public final class BeaconCompression {

    private static final int KEYS = 8;
    private static final int VALUES = 10;
    private static final int FILL = KEYS + VALUES;
    private static final int VOCAB = FILL + 1;
    private static final int DIM = 32;
    private static final int CHUNK = 8;

    private BeaconCompression() {
    }

    /**
     * One arm at one length.
     *
     * @param accuracy share answered correctly
     * @param memory memory entries the reader attends to
     */
    public record Cell(double accuracy, double memory) {
    }

    private record Batch(int[][] contexts, int[] keys, int[] answers) {
    }

    private static Batch batch(int size, int length, SplittableRandom r) {
        int[][] ctx = new int[size][length];
        int[] keys = new int[size];
        int[] answers = new int[size];
        for (int b = 0; b < size; b++) {
            java.util.Arrays.fill(ctx[b], FILL);
            int slots = length / 2;
            int[] where = shuffled(slots, r);
            int[] ks = shuffled(KEYS, r);
            int pairs = 4;
            for (int i = 0; i < pairs; i++) {
                ctx[b][2 * where[i]] = ks[i];
                ctx[b][2 * where[i] + 1] = KEYS + r.nextInt(VALUES);
            }
            int pick = r.nextInt(pairs);
            keys[b] = ks[pick];
            answers[b] = ctx[b][2 * where[pick] + 1] - KEYS;
        }
        return new Batch(ctx, keys, answers);
    }

    private static int[] shuffled(int n, SplittableRandom r) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return a;
    }

    /** The trainable encoder, reader and (for the beacon arms) compressor. */
    private static final class Arm {
        final String name;
        final Var embed;
        final Var previous;
        final Var wq;
        final Var wk;
        final Var wv;
        final Var w1;
        final Var w2;
        final BeaconCompressor compressor;

        Arm(String name, SplittableRandom r) {
            this.name = name;
            embed = param(r, 1f, VOCAB, DIM);
            previous = param(r, 1f, VOCAB, DIM);
            float s = (float) (1 / Math.sqrt(DIM));
            wq = param(r, s, DIM, DIM);
            wk = param(r, s, DIM, DIM);
            wv = param(r, s, DIM, DIM);
            w1 = param(r, s, 2 * DIM, 64);
            w2 = param(r, 0.125f, 64, VALUES);
            compressor = switch (name) {
                case "Activation Beacon 4x" -> BeaconCompressor.beacon(CHUNK, 2, DIM, r);
                case "Activation Beacon 8x" -> BeaconCompressor.beacon(CHUNK, 1, DIM, r);
                case "gist tokens" -> BeaconCompressor.gist(2, DIM, r);
                default -> null;
            };
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(embed, previous, wq, wk, wv, w1, w2));
            if (compressor != null) {
                p.addAll(compressor.parameters());
            }
            return p;
        }

        int memory(int length) {
            return switch (name) {
                case "full context" -> length;
                case "sliding window" -> CHUNK;
                case "mean pooling 4x (compressive)" -> length / 4;
                case "mean pooling 8x (compressive)" -> length / 8;
                default -> compressor.compressedLength(length);
            };
        }

        Var logits(Batch b) {
            int n = b.contexts()[0].length;
            int size = b.contexts().length;
            float[] cur = new float[size * n * VOCAB];
            float[] prev = new float[size * n * VOCAB];
            for (int i = 0; i < size; i++) {
                for (int t = 0; t < n; t++) {
                    cur[(i * n + t) * VOCAB + b.contexts()[i][t]] = 1;
                    prev[(i * n + t) * VOCAB + (t == 0 ? FILL : b.contexts()[i][t - 1])] = 1;
                }
            }
            Var x = Var.of(Tensor.of(Shape.of(size * n, VOCAB), cur)).matmul(embed)
                    .add(Var.of(Tensor.of(Shape.of(size * n, VOCAB), prev)).matmul(previous))
                    .reshape(size, n, DIM);
            Var mem = switch (name) {
                case "full context" -> x;
                case "sliding window" -> x.slice(1, n - CHUNK, n);
                case "mean pooling 4x (compressive)" -> x.reshape(size, n / 4, 4, DIM).mean(2, false);
                case "mean pooling 8x (compressive)" -> x.reshape(size, n / 8, 8, DIM).mean(2, false);
                default -> compressor.forward(x);
            };
            int m = mem.shape().dim(1);
            float[] qk = new float[size * VOCAB];
            for (int i = 0; i < size; i++) {
                qk[i * VOCAB + b.keys()[i]] = 1;
            }
            Var qe = Var.of(Tensor.of(Shape.of(size, VOCAB), qk)).matmul(embed);
            Var q = qe.matmul(wq).reshape(size, 1, 1, DIM);
            Var k = mem.matmul(wk).reshape(size, 1, m, DIM);
            Var v = mem.matmul(wv).reshape(size, 1, m, DIM);
            Var read = Functional.attention(q, k, v, null, (float) (1 / Math.sqrt(DIM)), 0, null).reshape(size, DIM);
            return Var.concat(1, qe, read).matmul(w1).tanh().matmul(w2);
        }
    }

    private static Var param(SplittableRandom r, float scale, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            f[i] = (float) (r.nextGaussian() * scale);
        }
        return Var.param(Tensor.of(Shape.of(dims), f));
    }

    /**
     * Trains every arm and tests it.
     *
     * @param steps training steps per arm
     * @param testLengths context lengths tested (multiples of 8)
     * @param questions test questions per length
     * @param seed the seed
     * @return arm to one cell per test length
     */
    public static Map<String, Cell[]> run(int steps, int[] testLengths, int questions, long seed) {
        Map<String, Cell[]> out = new LinkedHashMap<>();
        for (String name : List.of("full context", "sliding window", "mean pooling 4x (compressive)",
                "Activation Beacon 4x", "mean pooling 8x (compressive)", "Activation Beacon 8x", "gist tokens")) {
            Arm arm = new Arm(name, new SplittableRandom(seed));
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed + 1);
            for (int s = 0; s < steps; s++) {
                Batch b = batch(32, 32, r);
                Var loss = Functional.crossEntropy(arm.logits(b), ids(b.answers()));
                loss.backward();
                opt.step(arm.parameters());
            }
            Cell[] cells = new Cell[testLengths.length];
            for (int li = 0; li < testLengths.length; li++) {
                Batch b = batch(questions, testLengths[li], new SplittableRandom(seed + 100 + li));
                float[] logits = Var.noGrad(() -> arm.logits(b)).value().toArray();
                int ok = 0;
                for (int i = 0; i < questions; i++) {
                    int best = 0;
                    for (int c = 1; c < VALUES; c++) {
                        if (logits[i * VALUES + c] > logits[i * VALUES + best]) {
                            best = c;
                        }
                    }
                    ok += best == b.answers()[i] ? 1 : 0;
                }
                cells[li] = new Cell(ok / (double) questions, arm.memory(testLengths[li]));
            }
            out.put(name, cells);
        }
        return out;
    }

    private static Tensor ids(int[] a) {
        float[] f = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            f[i] = a[i];
        }
        return Tensor.of(Shape.of(a.length), f);
    }
}
