package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.Gptq;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * How many bits a language model can memorise per parameter. Small autoregressive models are
 * trained on uniformly random bit strings, where nothing can be generalised, so every bit of
 * compression is memorisation: the memorised information of a string is its length in bits
 * minus the model's code length {@code -log2 p(x)} (floored at 0), summed over the training
 * set. With more strings than the model can hold, the total saturates at its capacity;
 * dividing by the parameter count gives bits per parameter (about 3.6 for GPT-style models
 * trained to convergence in the paper). The same trained models are re-measured after
 * round-to-nearest weight quantisation to 8 and 4 bits, and for one size the memorised bits
 * are traced against the dataset size: they track the data's entropy until it exceeds the
 * capacity, then flatten, which is where the paper places the onset of double descent.
 *
 * <p>The models are next-bit MLPs over the whole prefix (one-hot bits plus a position
 * one-hot), the {@code WindowLm} shape, because attention decoders train too slowly on a CPU
 * for a capacity sweep; the measured bits per parameter are therefore those of a short
 * training budget, a lower bound on the converged capacity.
 *
 * <p>Morris et al. 2025, "How much do language models memorize?",
 * <a href="https://arxiv.org/abs/2505.24832">arXiv:2505.24832</a>.
 */
public final class MemorizationCapacity {

    /**
     * One model size.
     *
     * @param hidden the hidden width
     * @param parameters the parameter count
     * @param strings training strings
     * @param datasetBits the entropy of the training set
     * @param memorizedBits memorised bits at float32
     * @param bitsPerParameter {@code memorizedBits / parameters}
     * @param int8Bits memorised bits after 8-bit weight quantisation
     * @param int4Bits memorised bits after 4-bit weight quantisation
     */
    public record Size(int hidden, long parameters, int strings, double datasetBits, double memorizedBits,
            double bitsPerParameter, double int8Bits, double int4Bits) {
    }

    /**
     * One point of the dataset-size sweep.
     *
     * @param strings training strings
     * @param datasetBits the entropy of the training set
     * @param memorizedBits memorised bits
     */
    public record Point(int strings, double datasetBits, double memorizedBits) {
    }

    private MemorizationCapacity() {
    }

    private static Model model(int hidden, int length, long seed) {
        return Model.sequential().input(3 * length).dense(hidden, Activation.TANH).dense(hidden, Activation.TANH)
                .dense(2).seed(seed).build();
    }

    /** Every prefix as a row: one-hot bits so far, then the position; the target is the next bit. */
    private static Tensor[] rows(int[][] strings) {
        int length = strings[0].length;
        int n = strings.length * length;
        float[] x = new float[n * 3 * length];
        float[] y = new float[n];
        int row = 0;
        for (int[] s : strings) {
            for (int t = 0; t < length; t++) {
                int base = row * 3 * length;
                for (int j = 0; j < t; j++) {
                    x[base + 2 * j + s[j]] = 1;
                }
                x[base + 2 * length + t] = 1;
                y[row++] = s[t];
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 3 * length), x), Tensor.of(Shape.of(n), y)};
    }

    private static int[][] strings(int n, int length, SplittableRandom r) {
        int[][] s = new int[n][length];
        for (int[] row : s) {
            for (int t = 0; t < length; t++) {
                row[t] = r.nextInt(2);
            }
        }
        return s;
    }

    private static Model train(int hidden, int[][] data, int steps, long seed) {
        Model m = model(hidden, data[0].length, seed);
        Tensor[] xy = rows(data);
        Optimizer opt = Optimizer.adam(0.02f);
        for (int s = 0; s < steps; s++) {
            Var loss = Functional.crossEntropy(m.forward(Var.of(xy[0]), true), xy[1]);
            loss.backward();
            opt.step(m.parameters());
        }
        return m;
    }

    /**
     * Memorised bits: per string, its length minus its code length in bits, floored at 0.
     *
     * @param m a trained model
     * @param data the training strings
     * @return the summed memorised bits
     */
    static double memorized(Model m, int[][] data) {
        int length = data[0].length;
        Tensor[] xy = rows(data);
        float[] logits = m.forward(Var.of(xy[0])).value().to(DType.FLOAT32).toArray();
        double total = 0;
        for (int s = 0; s < data.length; s++) {
            double code = 0;
            for (int t = 0; t < length; t++) {
                int row = s * length + t;
                double a = logits[2 * row];
                double b = logits[2 * row + 1];
                double max = Math.max(a, b);
                double lse = max + Math.log(Math.exp(a - max) + Math.exp(b - max));
                code -= ((data[s][t] == 0 ? a : b) - lse) / Math.log(2);
            }
            total += Math.max(0, length - code);
        }
        return total;
    }

    private static Model quantized(Model base, int bits) {
        Model q = base.copy(base.device());
        for (Layer l : q.layers()) {
            if (l instanceof Dense d) {
                d.weight().assign(Gptq.roundToNearest(d.weight().value(), bits));
            }
        }
        return q;
    }

    /**
     * Measures model sizes, each trained on about four times as many bits as it has
     * parameters.
     *
     * @param hiddens hidden widths
     * @param length bits per string
     * @param steps full-batch training steps
     * @param seed the seed
     * @return one entry per size
     */
    public static List<Size> sizes(int[] hiddens, int length, int steps, long seed) {
        List<Size> out = new ArrayList<>();
        for (int hidden : hiddens) {
            long params = model(hidden, length, 0).parameterCount();
            int n = (int) Math.ceil(4.0 * params / length);
            int[][] data = strings(n, length, new SplittableRandom(seed + hidden));
            Model m = train(hidden, data, steps, seed);
            double mem = memorized(m, data);
            out.add(new Size(hidden, params, n, (double) n * length, mem, mem / params,
                    memorized(quantized(m, 8), data), memorized(quantized(m, 4), data)));
        }
        return out;
    }

    /**
     * Memorised bits of one model size against the number of training strings.
     *
     * @param hidden the hidden width
     * @param counts dataset sizes in strings
     * @param length bits per string
     * @param steps full-batch training steps
     * @param seed the seed
     * @return one point per dataset size
     */
    public static List<Point> datasetSweep(int hidden, int[] counts, int length, int steps, long seed) {
        List<Point> out = new ArrayList<>();
        for (int n : counts) {
            int[][] data = strings(n, length, new SplittableRandom(seed + n));
            out.add(new Point(n, (double) n * length, memorized(train(hidden, data, steps, seed), data)));
        }
        return out;
    }
}
