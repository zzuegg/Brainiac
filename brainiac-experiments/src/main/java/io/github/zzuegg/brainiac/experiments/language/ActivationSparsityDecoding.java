package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SparseActivationDense;
import io.github.zzuegg.brainiac.experiments.attention.ArchitectureBench;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Training-free activation sparsity (TEAL) for CPU decoding: a fixed-window MLP language
 * model (four previous tokens, two SiLU hidden layers of 512) is trained densely on the
 * {@link ArchitectureBench} toy language; then the input of every projection is thresholded
 * at the calibrated magnitude quantile for 0, 25, 40 and 50% sparsity
 * ({@link SparseActivationDense}). Reported: the validation loss (perplexity in nats), the
 * measured share of kept inputs, and single-token greedy decoding speed, where the sparse
 * path skips the weight rows of dropped inputs. The 0% row runs the same decode loop with
 * every row read, so the speed-up isolates the skipped rows.
 */
public final class ActivationSparsityDecoding {

    private static final int CONTEXT = 4;
    private static final int EMBED = 16;
    private static final int HIDDEN = 512;

    private ActivationSparsityDecoding() {
    }

    /**
     * One sparsity level's result.
     *
     * @param sparsity the target share of dropped inputs
     * @param validationLoss the next-token cross-entropy on validation windows
     * @param density the measured share of kept inputs while decoding
     * @param tokensPerSecond greedy decoding speed
     * @param speedup the speed relative to the dense row
     */
    public record Row(float sparsity, double validationLoss, double density, double tokensPerSecond,
            double speedup) {
    }

    /**
     * Trains the model and measures every sparsity level.
     *
     * @param sparsities the target sparsities, the first one 0 (the dense reference)
     * @param steps Adam steps on batches of 64 windows
     * @param decodeTokens tokens to decode for the speed measurement
     * @param seed the seed
     * @return one row per sparsity
     */
    public static List<Row> run(float[] sparsities, int steps, int decodeTokens, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int vocab = ArchitectureBench.VOCAB;
        Var embed = Var.param(Tensor.normal(Shape.of(vocab, EMBED), 0, 1, random));
        Dense[] layers = {new Dense(CONTEXT * EMBED, HIDDEN, Activation.SILU, Init.HE, random),
            new Dense(HIDDEN, HIDDEN, Activation.SILU, Init.HE, random),
            new Dense(HIDDEN, vocab, Activation.IDENTITY, Init.XAVIER, random)};
        List<Var> params = new ArrayList<>(List.of(embed));
        for (Dense d : layers) {
            params.addAll(d.parameters());
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            int[][] w = windows(random, 64);
            Functional.crossEntropy(forward(embed, List.of(layers[0], layers[1], layers[2]), w), targets(w))
                    .backward();
            opt.step(params);
        }
        // calibration inputs of every layer
        int[][] calib = windows(new SplittableRandom(seed + 5), 256);
        Tensor[] inputs = new Tensor[3];
        Var x = context(embed, calib);
        for (int l = 0; l < 3; l++) {
            inputs[l] = x.value();
            x = layers[l].forward(x);
        }
        int[][] val = windows(new SplittableRandom(seed + 9), 512);
        float[] table = embed.value().toArray();
        List<Row> rows = new ArrayList<>();
        double denseSpeed = 0;
        for (float sparsity : sparsities) {
            List<SparseActivationDense> sparse = new ArrayList<>();
            for (int l = 0; l < 3; l++) {
                sparse.add(SparseActivationDense.calibrated(layers[l], sparsity, inputs[l]));
            }
            double loss = Functional.crossEntropy(forward(embed, sparse, val), targets(val)).value().item();
            decode(table, sparse, Math.min(500, decodeTokens));                     // warm-up
            sparse.forEach(SparseActivationDense::resetCounts);
            long start = System.nanoTime();
            decode(table, sparse, decodeTokens);
            double seconds = (System.nanoTime() - start) / 1e9;
            double speed = decodeTokens / seconds;
            if (rows.isEmpty()) {
                denseSpeed = speed;
            }
            double density = sparse.stream().mapToDouble(SparseActivationDense::density).average().orElse(1);
            rows.add(new Row(sparsity, loss, density, speed, speed / denseSpeed));
        }
        return rows;
    }

    private static Var forward(Var embed, List<? extends io.github.zzuegg.brainiac.nn.Layer> layers, int[][] w) {
        Var x = context(embed, w);
        for (io.github.zzuegg.brainiac.nn.Layer l : layers) {
            x = l.forward(x);
        }
        return x;
    }

    private static Var context(Var embed, int[][] windows) {
        float[] ids = new float[windows.length * CONTEXT];
        for (int i = 0; i < windows.length; i++) {
            for (int j = 0; j < CONTEXT; j++) {
                ids[i * CONTEXT + j] = windows[i][j];
            }
        }
        return embed.gather(0, Tensor.of(Shape.of(ids.length), ids)).reshape(windows.length, CONTEXT * EMBED);
    }

    private static Tensor targets(int[][] windows) {
        float[] t = new float[windows.length];
        for (int i = 0; i < t.length; i++) {
            t[i] = windows[i][CONTEXT];
        }
        return Tensor.of(Shape.of(t.length), t);
    }

    /** Windows of {@code CONTEXT + 1} consecutive tokens from toy-language sequences. */
    private static int[][] windows(SplittableRandom random, int n) {
        int[][] seqs = ArchitectureBench.corpus(random, n);
        int[][] w = new int[n][CONTEXT + 1];
        for (int i = 0; i < n; i++) {
            int start = random.nextInt(seqs[i].length - CONTEXT);
            System.arraycopy(seqs[i], start, w[i], 0, CONTEXT + 1);
        }
        return w;
    }

    /** Greedy decoding on float arrays through the sparse fast path. */
    private static int decode(float[] table, List<SparseActivationDense> layers, int tokens) {
        int[] window = {1, 2, 3, 4};
        float[] in = new float[CONTEXT * EMBED];
        int checksum = 0;
        for (int n = 0; n < tokens; n++) {
            for (int j = 0; j < CONTEXT; j++) {
                System.arraycopy(table, window[j] * EMBED, in, j * EMBED, EMBED);
            }
            float[] h = in;
            for (SparseActivationDense l : layers) {
                h = l.decode(h);
            }
            int best = 0;
            for (int v = 1; v < h.length; v++) {
                best = h[v] > h[best] ? v : best;
            }
            System.arraycopy(window, 1, window, 0, CONTEXT - 1);
            window[CONTEXT - 1] = best == window[CONTEXT - 2] ? (best + n) % h.length : best;   // avoid fixed points
            checksum += best;
        }
        return checksum;
    }

    /**
     * Formats rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String table(List<Row> rows) {
        StringBuilder sb = new StringBuilder(String.format("%8s %9s %8s %10s %8s%n", "sparsity", "val loss",
                "density", "tokens/s", "speedup"));
        for (Row r : rows) {
            sb.append(String.format("%8.2f %9.4f %8.3f %10.0f %8.2f%n", r.sparsity(), r.validationLoss(), r.density(),
                    r.tokensPerSecond(), r.speedup()));
        }
        return sb.toString();
    }

    /**
     * Runs 0, 25, 40 and 50% sparsity.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(table(run(new float[] {0f, 0.25f, 0.4f, 0.5f}, 600, 20000, 1)));
    }
}
