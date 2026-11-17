package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.InputRecorder;
import io.github.zzuegg.brainiac.experimental.nn.SparseAutoencoder;
import io.github.zzuegg.brainiac.experimental.nn.Transcoder;
import io.github.zzuegg.brainiac.experiments.language.CountingTasks;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Transcoders against sparse autoencoders on MLPs. A transcoder predicts an MLP's output from
 * its input through sparse features, so its features describe the computation; an SAE on the
 * MLP output only compresses the result. At the same number of active features, how much
 * of the MLP output's variance does each explain? Two MLPs are measured: a random ReLU MLP
 * fed superposed sparse features (the {@link SuperpositionRecovery} toy), at several
 * sparsities (the reconstruction-vs-sparsity curve), and the feed-forward network of the
 * first block of a small decoder pretrained on counting sequences.
 *
 * <p>Dunefsky et al. 2024, "Transcoders Find Interpretable LLM Feature Circuits",
 * <a href="https://arxiv.org/abs/2406.11944">arXiv:2406.11944</a>.
 */
public final class TranscoderComparison {

    /**
     * One point of the curve.
     *
     * @param mlp which MLP ("superposition" or "toy LM")
     * @param k active features per sample
     * @param saeOnOutput explained variance of a top-k SAE on the MLP output
     * @param transcoder explained variance of a top-k transcoder from input to output
     * @param skipTranscoder the same with a linear skip
     */
    public record Point(String mlp, int k, double saeOnOutput, double transcoder, double skipTranscoder) {
    }

    private TranscoderComparison() {
    }

    /**
     * Runs both MLPs.
     *
     * @param ks the sparsities of the superposition curve
     * @param steps training steps per dictionary
     * @param seed the seed
     * @return the curve points, then the toy-LM point at the largest {@code k}
     */
    public static List<Point> run(int[] ks, int steps, long seed) {
        List<Point> out = new ArrayList<>();
        float[][][] toy = superpositionMlp(seed);
        for (int k : ks) {
            out.add(measure("superposition", toy[0], toy[1], k, 4 * toy[0][0].length, steps, seed));
        }
        float[][][] lm = decoderMlp(seed);
        out.add(measure("toy LM", lm[0], lm[1], ks[ks.length - 1], 2 * lm[0][0].length, steps, seed));
        return out;
    }

    private static Point measure(String name, float[][] x, float[][] y, int k, int features, int steps, long seed) {
        SparseAutoencoder sae = SparseAutoencoder.topK(y[0].length, features, k, seed + 10);
        sae.train(y, steps, 128, 3e-3f, seed + 11);
        Transcoder tc = Transcoder.of(x[0].length, y[0].length, features, k, false, seed + 10);
        tc.train(x, y, steps, 128, 3e-3f, seed + 11);
        Transcoder skip = Transcoder.of(x[0].length, y[0].length, features, k, true, seed + 10);
        skip.train(x, y, steps, 128, 3e-3f, seed + 11);
        return new Point(name, k, sae.explainedVariance(y), tc.explainedVariance(x, y), skip.explainedVariance(x, y));
    }

    /** Sparse superposed inputs through a random ReLU MLP: {@code {inputs, outputs}}. */
    static float[][][] superpositionMlp(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int truth = 24;
        int dim = 12;
        int hidden = 48;
        double[][] dirs = new double[truth][dim];
        for (double[] d : dirs) {
            double n = 0;
            for (int i = 0; i < dim; i++) {
                d[i] = r.nextGaussian();
                n += d[i] * d[i];
            }
            for (int i = 0; i < dim; i++) {
                d[i] /= Math.sqrt(n);
            }
        }
        double[][] w1 = new double[dim][hidden];
        double[][] w2 = new double[hidden][dim];
        for (double[] row : w1) {
            for (int j = 0; j < hidden; j++) {
                row[j] = r.nextGaussian() / Math.sqrt(dim);
            }
        }
        for (double[] row : w2) {
            for (int j = 0; j < dim; j++) {
                row[j] = r.nextGaussian() / Math.sqrt(hidden);
            }
        }
        int samples = 3000;
        float[][] x = new float[samples][dim];
        float[][] y = new float[samples][dim];
        for (int s = 0; s < samples; s++) {
            for (double[] d : dirs) {
                if (r.nextDouble() < 2.0 / truth) {
                    double a = 0.5 + r.nextDouble();
                    for (int i = 0; i < dim; i++) {
                        x[s][i] += (float) (a * d[i]);
                    }
                }
            }
            for (int h = 0; h < hidden; h++) {
                double pre = -0.2;
                for (int i = 0; i < dim; i++) {
                    pre += x[s][i] * w1[i][h];
                }
                if (pre > 0) {
                    for (int j = 0; j < dim; j++) {
                        y[s][j] += (float) (pre * w2[h][j]);
                    }
                }
            }
        }
        return new float[][][] {x, y};
    }

    /** The first block's feed-forward network of a pretrained counting decoder. */
    static float[][][] decoderMlp(long seed) {
        HookedDecoder m = CountingTasks.pretrained(150, seed).copy();
        Map<String, InputRecorder> rec = new HashMap<>();
        m.replaceLinears((name, l) -> {
            if (name.equals("block.0.ffn.up") || name.equals("block.0.ffn.down")) {
                InputRecorder ir = new InputRecorder(l);
                rec.put(name, ir);
                return ir;
            }
            return l;
        });
        m.forward(CountingTasks.sequences(new SplittableRandom(seed + 5), 160, 12, m.vocabSize(), 1, 2));
        Tensor in = rec.get("block.0.ffn.up").inputs();
        Tensor hid = rec.get("block.0.ffn.down").inputs();
        Layer down = rec.get("block.0.ffn.down").base();
        int dim = in.shape().dim(-1);
        int rows = (int) (in.shape().size() / dim);
        float[] x = in.reshape(rows, dim).to(DType.FLOAT32).toArray();
        float[] y = down.forward(Var.of(hid.reshape(rows, hid.shape().dim(-1)))).value().to(DType.FLOAT32).toArray();
        return new float[][][] {split(x, rows, dim), split(y, rows, dim)};
    }

    private static float[][] split(float[] flat, int rows, int dim) {
        float[][] out = new float[rows][dim];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * dim, out[i], 0, dim);
        }
        return out;
    }
}
