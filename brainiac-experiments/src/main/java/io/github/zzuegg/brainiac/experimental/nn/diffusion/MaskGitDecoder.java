package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * MaskGIT (Chang et al., <a href="https://arxiv.org/abs/2202.04200">2202.04200</a>):
 * parallel, iterative decoding of discrete image tokens with a bidirectional transformer.
 * Any {@link Model} mapping {@code [batch, length]} token ids to
 * {@code [batch, length, vocabulary]} logits (built without a causal mask) will do.
 *
 * <p><b>Training</b> ({@link #loss}): draw {@code r ~ U(0, 1)}, mask
 * {@code ceil(gamma(r) * length)} random tokens with the cosine schedule
 * {@code gamma(r) = cos(pi r / 2)}, and take the cross-entropy on the masked positions.
 *
 * <p><b>Decoding</b> ({@link #sample}): start fully masked; at step {@code t} of {@code T}
 * predict every masked token in parallel, sample them, and re-mask the
 * {@code floor(gamma((t + 1) / T) * length)} least confident ones, where confidence is the
 * sampled token's log-probability plus annealed Gumbel noise
 * {@code choiceTemperature * (1 - (t + 1) / T) * g}. Tokens once kept are never changed. An
 * image of {@code length} tokens takes {@code T << length} network calls, where an
 * autoregressive model takes {@code length}.
 */
public final class MaskGitDecoder {
    private final Model model;
    private final int vocabulary;
    private final int maskId;

    private MaskGitDecoder(Model model, int vocabulary, int maskId) {
        this.model = model;
        this.vocabulary = vocabulary;
        this.maskId = maskId;
    }

    /**
     * Wraps a bidirectional token model.
     *
     * @param model maps {@code [batch, length]} ids to {@code [batch, length, vocabulary]} logits
     * @param vocabulary the number of token ids, including the mask token
     * @param maskId the id of the mask token
     * @return the decoder
     */
    public static MaskGitDecoder of(Model model, int vocabulary, int maskId) {
        if (maskId < 0 || maskId >= vocabulary) {
            throw new IllegalArgumentException("mask id " + maskId + " is outside the vocabulary");
        }
        return new MaskGitDecoder(model, vocabulary, maskId);
    }

    /** @return the wrapped model */
    public Model model() {
        return model;
    }

    /**
     * The cosine mask schedule.
     *
     * @param r progress in [0, 1]
     * @return the share of tokens masked
     */
    public static double gamma(double r) {
        return Math.cos(Math.PI / 2 * r);
    }

    /**
     * The MaskGIT training loss on a batch.
     *
     * @param tokens clean token sequences, all of one length
     * @param random the source of mask ratios and masks
     * @return the mean cross-entropy over masked positions
     */
    public Var loss(int[][] tokens, RandomGenerator random) {
        int b = tokens.length;
        int len = tokens[0].length;
        float[] input = new float[b * len];
        float[] weight = new float[b * len * vocabulary];
        boolean[][] masked = new boolean[b][len];
        int total = 0;
        for (int i = 0; i < b; i++) {
            int k = Math.max(1, (int) Math.ceil(gamma(random.nextDouble()) * len));
            int[] order = shuffled(len, random);
            for (int m = 0; m < k; m++) {
                masked[i][order[m]] = true;
            }
            total += k;
        }
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < len; j++) {
                input[i * len + j] = masked[i][j] ? maskId : tokens[i][j];
                if (masked[i][j]) {
                    weight[(i * len + j) * vocabulary + tokens[i][j]] = 1f / total;
                }
            }
        }
        Var logits = model.forward(Var.of(Tensor.of(Shape.of(b, len), input)), true);
        Var logp = Activation.LOG_SOFTMAX.forward(logits.reshape(b * len, vocabulary));
        return logp.mul(Tensor.of(Shape.of(b * len, vocabulary), weight)).sum().neg();
    }

    /**
     * Generates sequences by iterative parallel decoding.
     *
     * @param count the number of sequences
     * @param length their length
     * @param steps the number of decoding steps {@code T}
     * @param temperature the sampling temperature of tokens
     * @param choiceTemperature the scale of the Gumbel noise on confidences (4.5 in the paper)
     * @param random the random source
     * @param trace receives the sequences after every step (mask ids included), or {@code null}
     * @return the generated sequences
     */
    public int[][] sample(int count, int length, int steps, float temperature, float choiceTemperature,
            RandomGenerator random, List<int[][]> trace) {
        int[][] x = new int[count][length];
        for (int[] row : x) {
            Arrays.fill(row, maskId);
        }
        for (int t = 0; t < steps; t++) {
            float[] flat = new float[count * length];
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < length; j++) {
                    flat[i * length + j] = x[i][j];
                }
            }
            float[] logits = model.forward(Var.of(Tensor.of(Shape.of(count, length), flat)), false).value().toArray();
            double progress = (t + 1.0) / steps;
            int keepMasked = t == steps - 1 ? 0 : (int) Math.floor(gamma(progress) * length);
            for (int i = 0; i < count; i++) {
                List<int[]> picks = new ArrayList<>();   // {position, token}
                List<Double> confidence = new ArrayList<>();
                int masked = 0;
                for (int j = 0; j < length; j++) {
                    if (x[i][j] != maskId) {
                        continue;
                    }
                    masked++;
                    double[] p = softmax(logits, (i * length + j) * vocabulary, temperature);
                    int token = draw(p, random);
                    double u = Math.max(1e-12, random.nextDouble());
                    double gumbel = -Math.log(-Math.log(u));
                    picks.add(new int[] {j, token});
                    confidence.add(Math.log(Math.max(1e-30, p[token])) + choiceTemperature * (1 - progress) * gumbel);
                }
                // reveal enough to leave keepMasked, but always at least one token per step
                int reveal = Math.max(Math.min(1, masked), masked - keepMasked);
                Integer[] order = new Integer[picks.size()];
                for (int k = 0; k < order.length; k++) {
                    order[k] = k;
                }
                Arrays.sort(order, (a, c) -> Double.compare(confidence.get(c), confidence.get(a)));
                for (int k = 0; k < reveal; k++) {
                    int[] pick = picks.get(order[k]);
                    x[i][pick[0]] = pick[1];
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

    private double[] softmax(float[] logits, int offset, float temperature) {
        double max = Double.NEGATIVE_INFINITY;
        for (int v = 0; v < vocabulary; v++) {
            if (v != maskId) {
                max = Math.max(max, logits[offset + v] / temperature);
            }
        }
        double[] p = new double[vocabulary];
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
        for (int v = 0; v < p.length; v++) {
            u -= p[v];
            if (u <= 0) {
                return v;
            }
        }
        int last = p.length - 1;
        while (last > 0 && p[last] == 0) {
            last--;
        }
        return last;
    }

    private static int[] shuffled(int n, RandomGenerator random) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return a;
    }
}
