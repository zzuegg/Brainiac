package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Discrete flow matching (Gat et al., <a href="https://arxiv.org/abs/2407.15595">arXiv:2407.15595</a>):
 * a continuous-time Markov chain over token sequences that moves probability from a source
 * distribution at {@code t = 0} to the data at {@code t = 1} along the mixture path
 * {@code p_t(x | x1) = kappa(t) delta_{x1} + (1 - kappa(t)) p_0}, independently per position.
 * A bidirectional token model (as in {@link MaskedDiffusion}) learns the posterior
 * {@code p(x1 | x_t)} by cross-entropy; sampling follows the probability velocity
 * {@code u_t(y | x_t) = kappa'/(1 - kappa) [p(x1 = y | x_t) - delta_{x_t}(y)]} with Euler jumps.
 * <ul>
 *   <li>{@link Source#MASK}: the source is all-mask, which recovers masked diffusion, but
 *       tokens are revealed by the velocity's jump probabilities, not by a confidence
 *       ranking;</li>
 *   <li>{@link Source#UNIFORM}: the source is uniform noise over the real tokens, so every
 *       position can change several times while sampling (masked diffusion can never revise a
 *       revealed token);</li>
 *   <li>a <b>corrector</b> adds the backward velocity {@code alpha kappa'/kappa [delta_{x_t} -
 *       p_0]} and scales the forward one by {@code 1 + alpha}, so the chain keeps re-noising
 *       and re-predicting some tokens at the same marginals: it fixes early mistakes.</li>
 * </ul>
 * The scheduler is {@code kappa(t) = t^power}.
 */
public final class DiscreteFlowMatching {

    /** The source distribution at {@code t = 0}. */
    public enum Source {
        /** Every position starts as the mask token. */
        MASK,
        /** Every position starts as a uniformly random real token. */
        UNIFORM
    }

    private final Model model;
    private final int vocabulary;
    private final int maskId;
    private final Source source;
    private final float power;

    private DiscreteFlowMatching(Model model, int vocabulary, int maskId, Source source, float power) {
        this.model = model;
        this.vocabulary = vocabulary;
        this.maskId = maskId;
        this.source = source;
        this.power = power;
    }

    /**
     * Wraps a bidirectional token model.
     *
     * @param model maps {@code [batch, length]} ids to {@code [batch, length, vocabulary]} logits
     * @param vocabulary the number of ids, including the mask id
     * @param maskId the mask id, which the model never predicts; it is still reserved under {@link Source#UNIFORM}
     * @param source the source distribution
     * @param power the scheduler exponent: {@code kappa(t) = t^power}, 1 for linear
     * @return the model
     */
    public static DiscreteFlowMatching of(Model model, int vocabulary, int maskId, Source source, float power) {
        if (maskId < 0 || maskId >= vocabulary) {
            throw new IllegalArgumentException("mask id " + maskId + " is outside the vocabulary");
        }
        return new DiscreteFlowMatching(model, vocabulary, maskId, source, power);
    }

    /** @return the wrapped model */
    public Model model() {
        return model;
    }

    private double kappa(double t) {
        return Math.pow(t, power);
    }

    private double kappaRate(double t) {
        return power * Math.pow(t, power - 1);
    }

    private int sourceToken(RandomGenerator random) {
        if (source == Source.MASK) {
            return maskId;
        }
        int k = random.nextInt(vocabulary - 1);
        return k >= maskId ? k + 1 : k;
    }

    /**
     * The training loss: corrupt each sequence along the path at a random {@code t} and score
     * the cross-entropy of the clean token at every corrupted position (every position for a
     * uniform source, where corruption is invisible).
     *
     * @param tokens clean sequences of one length
     * @param random the source of times and corruption
     * @return the scalar loss
     */
    public Var loss(int[][] tokens, RandomGenerator random) {
        int b = tokens.length;
        int len = tokens[0].length;
        float[] input = new float[b * len];
        float[] weight = new float[b * len * vocabulary];
        int scored = 0;
        boolean[][] use = new boolean[b][len];
        for (int i = 0; i < b; i++) {
            double k = kappa(random.nextDouble());
            for (int j = 0; j < len; j++) {
                boolean keep = random.nextDouble() < k;
                input[i * len + j] = keep ? tokens[i][j] : sourceToken(random);
                use[i][j] = source == Source.UNIFORM || !keep;
                scored += use[i][j] ? 1 : 0;
            }
        }
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < len; j++) {
                if (use[i][j]) {
                    weight[(i * len + j) * vocabulary + tokens[i][j]] = 1f / Math.max(1, scored);
                }
            }
        }
        Var logits = model.forward(Var.of(Tensor.of(Shape.of(b, len), input)), true);
        Var logp = Activation.LOG_SOFTMAX.forward(logits.reshape(b * len, vocabulary));
        return logp.mul(Tensor.of(Shape.of(b * len, vocabulary), weight)).sum().neg();
    }

    /**
     * Generates sequences with Euler jumps of the probability velocity.
     *
     * @param count how many sequences
     * @param length their length
     * @param steps the number of Euler steps (network evaluations)
     * @param corrector the corrector strength {@code alpha}, 0 for none; applied for
     *     {@code t} in [0.1, 0.8]
     * @param temperature the softmax temperature of the posterior
     * @param random the random source
     * @param trace receives the sequences after every step, or {@code null}
     * @return the sequences
     */
    public int[][] sample(int count, int length, int steps, float corrector, float temperature,
            RandomGenerator random, List<int[][]> trace) {
        int[][] x = new int[count][length];
        for (int[] row : x) {
            for (int j = 0; j < length; j++) {
                row[j] = sourceToken(random);
            }
        }
        double h = 1.0 / steps;
        for (int s = 0; s < steps; s++) {
            double t = s * h;
            float[] flat = new float[count * length];
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < length; j++) {
                    flat[i * length + j] = x[i][j];
                }
            }
            float[] logits = model.forward(Var.of(Tensor.of(Shape.of(count, length), flat)), false).value().toArray();
            double rate = kappaRate(t) / Math.max(1e-6, 1 - kappa(t));
            boolean correct = corrector > 0 && t >= 0.1 && t <= 0.8;
            double forward = h * rate * (correct ? 1 + corrector : 1);
            double backward = correct ? h * corrector * kappaRate(t) / Math.max(1e-6, kappa(t)) : 0;
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < length; j++) {
                    // under a mask source a revealed token is already a sample of x1: only the
                    // corrector moves it
                    boolean open = source == Source.UNIFORM || x[i][j] == maskId;
                    double u = random.nextDouble();
                    if (open && (s == steps - 1 || u < forward)) {
                        // jump towards the posterior (possibly to the same token)
                        x[i][j] = draw(posterior(logits, (i * length + j) * vocabulary, temperature), random);
                    } else if (open ? u < forward + backward : u < backward) {
                        x[i][j] = sourceToken(random);   // corrector: back towards the source
                    }
                }
            }
            if (trace != null) {
                int[][] copy = new int[count][];
                for (int i = 0; i < count; i++) {
                    copy[i] = x[i].clone();
                }
                trace.add(copy);
            }
        }
        return x;
    }

    private double[] posterior(float[] logits, int offset, float temperature) {
        double[] p = new double[vocabulary];
        double max = Double.NEGATIVE_INFINITY;
        for (int v = 0; v < vocabulary; v++) {
            if (v != maskId) {
                max = Math.max(max, logits[offset + v] / temperature);
            }
        }
        double sum = 0;
        for (int v = 0; v < vocabulary; v++) {
            p[v] = v == maskId ? 0 : Math.exp(logits[offset + v] / temperature - max);
            sum += p[v];
        }
        for (int v = 0; v < vocabulary; v++) {
            p[v] /= sum;
        }
        return p;
    }

    private static int draw(double[] p, RandomGenerator random) {
        double u = random.nextDouble();
        double c = 0;
        for (int v = 0; v < p.length; v++) {
            c += p[v];
            if (u < c) {
                return v;
            }
        }
        int last = p.length - 1;
        while (last > 0 && p[last] == 0) {
            last--;
        }
        return last;
    }

    /**
     * The source token of a position, for tests and tracing.
     *
     * @return the mask id under {@link Source#MASK}, else -1
     */
    public int sourceMarker() {
        return source == Source.MASK ? maskId : -1;
    }

    @Override
    public String toString() {
        return "DiscreteFlowMatching" + Arrays.asList(source, power);
    }
}
