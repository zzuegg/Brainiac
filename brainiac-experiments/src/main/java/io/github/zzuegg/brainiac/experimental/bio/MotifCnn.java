package io.github.zzuegg.brainiac.experimental.bio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A DeepBind-style sequence classifier: one-hot residues, a 1-D convolution with {@code F}
 * filters of width {@code W} (the motif detectors), ReLU, global max-pooling over positions and
 * a linear read-out to a bound/unbound logit. After training, each filter is read back as a
 * position weight matrix by counting the residues of the windows that activate it strongly
 * (above half its maximum), as in Alipanahi et al., "Predicting the sequence specificities of
 * DNA- and RNA-binding proteins by deep learning" (Nature Biotechnology 2015).
 *
 * <p>As a {@link Layer} it maps flattened one-hot input {@code [batch, length * alphabet]} to
 * logits {@code [batch, 1]}, so it drops into a model as a scoring head.
 */
public final class MotifCnn implements Layer {

    private final int alphabet;
    private final int length;
    private final int width;
    private final int filters;
    private final Var conv;
    private final Var convBias;
    private final Var out;
    private final Var outBias;

    /**
     * Creates an untrained classifier.
     *
     * @param alphabet the alphabet size
     * @param length the (fixed) sequence length
     * @param width the filter width
     * @param filters the number of filters
     * @param seed the seed
     */
    public MotifCnn(int alphabet, int length, int width, int filters, long seed) {
        this.alphabet = alphabet;
        this.length = length;
        this.width = width;
        this.filters = filters;
        SplittableRandom rnd = new SplittableRandom(seed);
        float[] c = new float[width * alphabet * filters];
        for (int i = 0; i < c.length; i++) {
            c[i] = (float) ((rnd.nextDouble() - 0.5) * 0.5);
        }
        float[] o = new float[filters];
        for (int i = 0; i < o.length; i++) {
            o[i] = (float) ((rnd.nextDouble() - 0.5) * 0.5);
        }
        conv = Var.param(Tensor.of(Shape.of(width * alphabet, filters), c));
        convBias = Var.param(Tensor.zeros(Shape.of(filters)));
        out = Var.param(Tensor.of(Shape.of(filters, 1), o));
        outBias = Var.param(Tensor.zeros(Shape.of(1)));
    }

    /**
     * One-hot encodes sequences (padding or truncating to the model length).
     *
     * @param seqs the sequences
     * @return {@code [batch, length * alphabet]}
     */
    public Tensor oneHot(int[][] seqs) {
        float[] x = new float[seqs.length * length * alphabet];
        for (int b = 0; b < seqs.length; b++) {
            for (int p = 0; p < Math.min(length, seqs[b].length); p++) {
                x[(b * length + p) * alphabet + seqs[b][p]] = 1;
            }
        }
        return Tensor.of(Shape.of(seqs.length, length * alphabet), x);
    }

    private Var activations(Var input) {
        int batch = input.shape().dim(0);
        int positions = length - width + 1;
        Var x = input.reshape(batch, length, alphabet);
        Var acc = null;
        for (int w = 0; w < width; w++) {
            Var part = x.slice(1, w, w + positions).reshape(batch * positions, alphabet)
                    .matmul(conv.slice(0, w * alphabet, (w + 1) * alphabet));
            acc = acc == null ? part : acc.add(part);
        }
        return acc.add(convBias).relu().reshape(batch, positions, filters);
    }

    @Override
    public Var forward(Var input) {
        return activations(input).max(1, false).matmul(out).add(outBias);
    }

    @Override
    public List<Var> parameters() {
        return List.of(conv, convBias, out, outBias);
    }

    @Override
    public int outputSize(int inputSize) {
        return 1;
    }

    /**
     * Trains by Adam on the logistic loss with mini-batches.
     *
     * @param seqs sequences
     * @param labels bound (true) or not
     * @param steps gradient steps
     * @param batch the batch size
     * @param learningRate the Adam step size
     * @param seed the batch-sampling seed
     * @return the final mini-batch loss
     */
    public double fit(int[][] seqs, boolean[] labels, int steps, int batch, float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom rnd = new SplittableRandom(seed);
        double last = 0;
        for (int s = 0; s < steps; s++) {
            int[][] xb = new int[batch][];
            float[] yb = new float[batch];
            for (int b = 0; b < batch; b++) {
                int i = rnd.nextInt(seqs.length);
                xb[b] = seqs[i];
                yb[b] = labels[i] ? 1 : 0;
            }
            Var z = forward(Var.of(oneHot(xb)));
            Var y = Var.of(Tensor.of(Shape.of(batch, 1), yb));
            Var loss = z.softplus().sub(z.mul(y)).mean();
            loss.backward();
            last = loss.value().item();
            opt.step(parameters());
        }
        return last;
    }

    /**
     * The bound logits.
     *
     * @param seqs sequences
     * @return one score per sequence
     */
    public double[] scores(int[][] seqs) {
        float[] z = Var.noGrad(() -> forward(Var.of(oneHot(seqs)))).value().toArray();
        double[] out = new double[seqs.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = z[i];
        }
        return out;
    }

    /**
     * Reads every filter back as a PWM from the windows that activate it above half its
     * maximum activation over the given sequences.
     *
     * @param seqs sequences (usually the positives)
     * @return {@code [filter][width][alphabet]}
     */
    public double[][][] filterPwms(int[][] seqs) {
        int positions = length - width + 1;
        float[] act = Var.noGrad(() -> activations(Var.of(oneHot(seqs)))).value().toArray();
        double[][][] pwm = new double[filters][width][alphabet];
        for (int f = 0; f < filters; f++) {
            double max = 0;
            for (int b = 0; b < seqs.length; b++) {
                for (int p = 0; p < positions; p++) {
                    max = Math.max(max, act[(b * positions + p) * filters + f]);
                }
            }
            for (int b = 0; b < seqs.length; b++) {
                for (int p = 0; p < positions; p++) {
                    if (max > 0 && act[(b * positions + p) * filters + f] > 0.5 * max && p + width <= seqs[b].length) {
                        for (int w = 0; w < width; w++) {
                            pwm[f][w][seqs[b][p + w]] += 1;
                        }
                    }
                }
            }
            for (int w = 0; w < width; w++) {
                double tot = alphabet * 0.1;
                for (int a = 0; a < alphabet; a++) {
                    tot += pwm[f][w][a];
                }
                for (int a = 0; a < alphabet; a++) {
                    pwm[f][w][a] = (pwm[f][w][a] + 0.1) / tot;
                }
            }
        }
        return pwm;
    }

    /** @return the filter with the largest positive read-out weight (the main "bound" detector) */
    public int topFilter() {
        float[] w = out.value().toArray();
        int best = 0;
        for (int f = 1; f < w.length; f++) {
            if (w[f] > w[best]) {
                best = f;
            }
        }
        return best;
    }
}
