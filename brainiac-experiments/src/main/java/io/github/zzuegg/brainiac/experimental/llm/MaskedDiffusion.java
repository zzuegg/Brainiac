package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Masked diffusion for text, as in LLaDA (Nie et al., <a href="https://arxiv.org/abs/2502.09992">2502.09992</a>)
 * and MDLM (<a href="https://arxiv.org/abs/2406.07524">2406.07524</a>): a bidirectional
 * transformer (any {@link Model} mapping {@code [batch, length]} token ids to
 * {@code [batch, length, vocabulary]} logits, built without a causal mask) learns to fill
 * in masked tokens.
 *
 * <p><b>Training</b> ({@link #loss}): draw a mask ratio {@code t ~ U(0, 1)} per sequence
 * (here in its discrete, lower-variance form: mask exactly {@code k ~ U{1..length}} random
 * positions, {@code t = k / length}), replace those tokens by the mask token, and score the
 * cross-entropy on the masked positions only, weighted by {@code 1 / t}. This is an upper
 * bound on the negative log-likelihood, so the model is a proper generative model.
 *
 * <p><b>Sampling</b> ({@link #sample}): start from an all-mask sequence and, over a fixed
 * number of steps, predict every masked token in parallel, keep the most confident
 * predictions and leave the rest masked ("low-confidence remasking"). The returned trace
 * shows the text crystallizing step by step. Tokens are chosen with a core {@link Sampling}.
 */
public final class MaskedDiffusion {
    private final Model model;
    private final int vocabulary;
    private final int maskId;

    private MaskedDiffusion(Model model, int vocabulary, int maskId) {
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
     * @return the masked diffusion model
     */
    public static MaskedDiffusion of(Model model, int vocabulary, int maskId) {
        if (maskId < 0 || maskId >= vocabulary) {
            throw new IllegalArgumentException("mask id " + maskId + " is outside the vocabulary");
        }
        return new MaskedDiffusion(model, vocabulary, maskId);
    }

    /** @return the wrapped model */
    public Model model() {
        return model;
    }

    /**
     * The LLaDA training loss on a batch.
     *
     * @param tokens clean sequences, all of one length
     * @param random the source of mask ratios and masks
     * @return the scalar loss, ready for {@code backward()}
     */
    public Var loss(int[][] tokens, RandomGenerator random) {
        int b = tokens.length;
        int len = tokens[0].length;
        float[] input = new float[b * len];
        float[] weight = new float[b * len * vocabulary];
        for (int i = 0; i < b; i++) {
            // mask exactly k ~ U{1..len} positions: the discrete form of t ~ U(0, 1), t = k / len,
            // with lower variance than independent coin flips
            int k = 1 + random.nextInt(len);
            int[] order = shuffled(len, random);
            boolean[] masked = new boolean[len];
            for (int m = 0; m < k; m++) {
                masked[order[m]] = true;
            }
            for (int j = 0; j < len; j++) {
                input[i * len + j] = masked[j] ? maskId : tokens[i][j];
                if (masked[j]) {
                    // one-hot of the true token, weighted by 1/t = len/k and averaged over positions
                    weight[(i * len + j) * vocabulary + tokens[i][j]] = (float) ((double) len / k / (len * b));
                }
            }
        }
        Var logits = model.forward(Var.of(Tensor.of(Shape.of(b, len), input)), true);
        Var logp = Activation.LOG_SOFTMAX.forward(logits.reshape(b * len, vocabulary));
        return logp.mul(Tensor.of(Shape.of(b * len, vocabulary), weight)).sum().neg();
    }

    /**
     * Generates sequences by iterative parallel unmasking.
     *
     * @param count the number of sequences
     * @param length their length
     * @param steps the number of unmasking steps (at most {@code length})
     * @param prompt fixed tokens per position, or -1 for positions to generate; {@code null} for none
     * @param sampling how each token is chosen from its distribution
     * @param random the random source
     * @param trace receives, after every step, the current sequences (mask tokens included), or {@code null}
     * @return the generated sequences
     */
    public int[][] sample(int count, int length, int steps, int[] prompt, Sampling sampling, RandomGenerator random,
            List<int[][]> trace) {
        int[][] x = new int[count][length];
        for (int[] row : x) {
            Arrays.fill(row, maskId);
            if (prompt != null) {
                for (int j = 0; j < length; j++) {
                    if (prompt[j] >= 0) {
                        row[j] = prompt[j];
                    }
                }
            }
        }
        for (int s = 1; s <= steps; s++) {
            float[] flat = new float[count * length];
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < length; j++) {
                    flat[i * length + j] = x[i][j];
                }
            }
            float[] logits = model.forward(Var.of(Tensor.of(Shape.of(count, length), flat)), false).value().toArray();
            for (int i = 0; i < count; i++) {
                List<int[]> candidates = new ArrayList<>();   // {position, token}
                List<Double> confidence = new ArrayList<>();
                int masked = 0;
                for (int j = 0; j < length; j++) {
                    if (x[i][j] != maskId) {
                        continue;
                    }
                    masked++;
                    float[] row = Arrays.copyOfRange(logits, (i * length + j) * vocabulary,
                            (i * length + j + 1) * vocabulary);
                    row[maskId] = Float.NEGATIVE_INFINITY;   // never predict the mask itself
                    int token = sampling.sample(row, new int[0], random);
                    candidates.add(new int[] {j, token});
                    confidence.add(softmax(row)[token]);
                }
                // after step s, round(length * (1 - s/steps)) of the generated positions stay masked
                int keepMasked = (int) Math.round((double) masked * (steps - s) / (steps - s + 1));
                int reveal = masked - keepMasked;
                Integer[] order = new Integer[candidates.size()];
                for (int k = 0; k < order.length; k++) {
                    order[k] = k;
                }
                Arrays.sort(order, (a, c) -> Double.compare(confidence.get(c), confidence.get(a)));
                for (int k = 0; k < reveal; k++) {
                    int[] pick = candidates.get(order[k]);
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

    private static double[] softmax(float[] row) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : row) {
            max = Math.max(max, v);
        }
        double[] p = new double[row.length];
        double sum = 0;
        for (int i = 0; i < row.length; i++) {
            p[i] = Math.exp(row[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < row.length; i++) {
            p[i] /= sum;
        }
        return p;
    }
}
