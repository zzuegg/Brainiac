package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.HiddenMarkovModel;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A linear-chain conditional random field output layer (Lafferty, McCallum and Pereira 2001,
 * <a href="https://repository.upenn.edu/cis_papers/159/">Conditional random fields</a>; with a
 * BiLSTM encoder, Huang, Xu and Yu 2015, <a href="https://arxiv.org/abs/1508.01991">arXiv:1508.01991</a>).
 * A path of tags {@code y} through a sequence scores
 * {@code sum_t emit[t, y_t] + sum_t trans[y_t-1, y_t] + start[y_0] + end[y_T-1]}, and
 * {@code p(y | x)} normalises that score over all paths with the forward algorithm (a
 * log-sum-exp recursion written in {@link Var} ops, so gradients flow into the transitions and
 * the encoder that produced the emissions). Decoding is exact Viterbi.
 *
 * <p>Why it matters: per-token softmax predicts each tag independently and can output
 * impossible sequences (an {@code I-} tag after {@code O}); the CRF learns that such transitions
 * never occur and decodes the best whole sequence.
 *
 * <p>As a {@link Layer} it passes emission scores {@code [batch, steps, tags]} through unchanged
 * and contributes its transition parameters to the model, so it drops into
 * {@code Model.sequential().add(LinearChainCrf.spec(tags))} with {@link #loss()} as the loss.
 */
public final class LinearChainCrf implements Layer {

    private final int tags;
    private final Var transitions;
    private final Var start;
    private final Var end;

    /**
     * Creates the layer with small random transition scores.
     *
     * @param tags the number of tags
     * @param random initialises the transitions
     */
    public LinearChainCrf(int tags, SplittableRandom random) {
        this.tags = tags;
        float[] t = new float[tags * tags];
        for (int i = 0; i < t.length; i++) {
            t[i] = (float) (0.01 * random.nextGaussian());
        }
        this.transitions = Var.param(Tensor.of(Shape.of(tags, tags), t));
        this.start = Var.param(Tensor.zeros(Shape.of(tags)));
        this.end = Var.param(Tensor.zeros(Shape.of(tags)));
    }

    /**
     * A spec for sequential models: emissions {@code [steps, tags]} per example pass through.
     *
     * @param tags the number of tags
     * @return the spec
     */
    public static LayerSpec spec(int tags) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new LinearChainCrf(tags, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return input;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public List<Var> parameters() {
        return List.of(transitions, start, end);
    }

    /** log-sum-exp over an axis, stabilised by the (constant) maximum. */
    private static Var logSumExp(Var x, int axis) {
        Tensor m = x.value().max(axis, true);
        Var s = x.sub(m).exp().sum(axis, true).log().add(m);
        int[] dims = s.shape().toArray();
        int[] squeezed = new int[dims.length - 1];
        for (int i = 0, j = 0; i < dims.length; i++) {
            if (i != (axis < 0 ? dims.length + axis : axis)) {
                squeezed[j++] = dims[i];
            }
        }
        return s.reshape(squeezed);
    }

    /**
     * The mean negative log-likelihood of gold tag sequences.
     *
     * @param emissions scores {@code [batch, steps, tags]}
     * @param gold tag ids {@code [batch, steps]}
     * @return a scalar
     */
    public Var negativeLogLikelihood(Var emissions, Tensor gold) {
        Shape s = emissions.shape();
        int b = s.dim(0);
        int steps = s.dim(1);
        float[] g = gold.toArray();
        float[] oneHot = new float[b * steps * tags];
        float[] counts = new float[tags * tags];
        float[] first = new float[tags];
        float[] last = new float[tags];
        for (int i = 0; i < b; i++) {
            for (int t = 0; t < steps; t++) {
                int y = (int) g[i * steps + t];
                oneHot[(i * steps + t) * tags + y] = 1;
                if (t > 0) {
                    counts[(int) g[i * steps + t - 1] * tags + y] += 1;
                }
            }
            first[(int) g[i * steps]] += 1;
            last[(int) g[i * steps + steps - 1]] += 1;
        }
        Var goldScore = emissions.mul(Tensor.of(s, oneHot)).sum()
                .add(transitions.mul(Tensor.of(Shape.of(tags, tags), counts)).sum())
                .add(start.mul(Tensor.of(Shape.of(tags), first)).sum())
                .add(end.mul(Tensor.of(Shape.of(tags), last)).sum());
        Var alpha = emissions.slice(1, 0, 1).reshape(b, tags).add(start);
        Var trans = transitions.reshape(1, tags, tags);
        for (int t = 1; t < steps; t++) {
            Var scores = alpha.reshape(b, tags, 1).add(trans);   // [b, from, to]
            alpha = logSumExp(scores, 1).add(emissions.slice(1, t, t + 1).reshape(b, tags));
        }
        Var logZ = logSumExp(alpha.add(end), 1).sum();
        return logZ.sub(goldScore).mul(1f / b);
    }

    /**
     * The negative log-likelihood as a core {@link Loss}: predictions are emissions
     * {@code [batch, steps, tags]}, targets tag ids {@code [batch, steps]}.
     *
     * @return the loss
     */
    public Loss loss() {
        return this::negativeLogLikelihood;
    }

    /**
     * Viterbi-decodes the best tag path of each sequence.
     *
     * @param emissions scores {@code [batch, steps, tags]}
     * @return {@code [batch][steps]} tag ids
     */
    public int[][] decode(Tensor emissions) {
        int b = emissions.shape().dim(0);
        int steps = emissions.shape().dim(1);
        float[] e = emissions.toArray();
        double[][] tr = transitionScores();
        float[] st = start.value().toArray();
        float[] en = end.value().toArray();
        double[] logStart = new double[tags];
        for (int j = 0; j < tags; j++) {
            logStart[j] = st[j];
        }
        int[][] out = new int[b][];
        for (int i = 0; i < b; i++) {
            double[][] emit = new double[steps][tags];
            for (int t = 0; t < steps; t++) {
                for (int j = 0; j < tags; j++) {
                    emit[t][j] = e[(i * steps + t) * tags + j] + (t == steps - 1 ? en[j] : 0);
                }
            }
            out[i] = HiddenMarkovModel.viterbi(logStart, tr, emit);
        }
        return out;
    }

    /**
     * Returns the learned transition scores.
     *
     * @return {@code [from][to]}
     */
    public double[][] transitionScores() {
        float[] t = transitions.value().toArray();
        double[][] tr = new double[tags][tags];
        for (int i = 0; i < tags; i++) {
            for (int j = 0; j < tags; j++) {
                tr[i][j] = t[i * tags + j];
            }
        }
        return tr;
    }
}
