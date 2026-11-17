package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Connectionist Temporal Classification (Graves et al., 2006,
 * <a href="https://www.cs.toronto.edu/~graves/icml_2006.pdf">ICML</a>): a loss for
 * sequence-to-sequence problems where the output is shorter than the input and nobody says
 * which input frame produced which output symbol, such as speech recognition or
 * handwriting. The network emits, per frame, a distribution over the symbols plus a
 * <em>blank</em> (class 0). An alignment maps frames to a label sequence by merging repeats
 * and dropping blanks; CTC sums the probability of every alignment of the target with the
 * forward-backward algorithm (in log space), and the loss is its negative log.
 *
 * <p>The gradient with respect to the logits is hand-derived
 * ({@code softmax - posterior occupancy of each class}) and checked by
 * {@code GradCheck}. As a {@link Loss}, predictions are logits
 * {@code [batch, frames, classes]} and targets are {@code [batch, maxLength]} label ids
 * {@code 1..classes-1}, padded with -1.
 */
public final class CtcLoss implements Loss {
    /** The shared instance; CTC has no settings besides the blank, which is class 0. */
    public static final CtcLoss INSTANCE = new CtcLoss();

    private CtcLoss() {
    }

    @Override
    public Var compute(Var prediction, Tensor target) {
        Shape s = prediction.shape();
        if (s.rank() != 3 || target.shape().rank() != 2 || target.shape().dim(0) != s.dim(0)) {
            throw new IllegalArgumentException("CTC takes logits [batch, frames, classes] and labels [batch, length],"
                    + " got " + s + " and " + target.shape());
        }
        int b = s.dim(0);
        int len = target.shape().dim(1);
        float[] t = target.toArray();
        int[][] labels = new int[b][];
        for (int i = 0; i < b; i++) {
            List<Integer> l = new ArrayList<>();
            for (int j = 0; j < len; j++) {
                int v = Math.round(t[i * len + j]);
                if (v >= 0) {
                    l.add(v);
                }
            }
            labels[i] = l.stream().mapToInt(Integer::intValue).toArray();
        }
        return loss(prediction, labels);
    }

    /**
     * The mean CTC loss over a batch.
     *
     * @param logits {@code [batch, frames, classes]}
     * @param labels one label sequence per batch element, ids {@code 1..classes-1}
     * @return the scalar loss
     */
    public static Var loss(Var logits, int[][] labels) {
        return Var.apply(new CtcOp(labels), logits);
    }

    /**
     * The operation itself, for {@code GradCheck}.
     *
     * @param labels one label sequence per batch element
     * @return the operation
     */
    public static Op op(int[][] labels) {
        return new CtcOp(labels);
    }

    /**
     * Best-path decoding: the most likely class per frame, repeats merged, blanks dropped.
     *
     * @param logits {@code [batch, frames, classes]}
     * @return the decoded label sequence per batch element
     */
    public static int[][] greedyDecode(Tensor logits) {
        int b = logits.shape().dim(0);
        int frames = logits.shape().dim(1);
        int c = logits.shape().dim(2);
        float[] v = logits.toArray();
        int[][] out = new int[b][];
        for (int i = 0; i < b; i++) {
            List<Integer> seq = new ArrayList<>();
            int prev = -1;
            for (int f = 0; f < frames; f++) {
                int best = 0;
                for (int k = 1; k < c; k++) {
                    if (v[(i * frames + f) * c + k] > v[(i * frames + f) * c + best]) {
                        best = k;
                    }
                }
                if (best != prev && best != 0) {
                    seq.add(best);
                }
                prev = best;
            }
            out[i] = seq.stream().mapToInt(Integer::intValue).toArray();
        }
        return out;
    }

    private static final class CtcOp implements Op {
        private final int[][] labels;
        private double[][][] gradients;   // d loss_i / d logits, before averaging

        CtcOp(int[][] labels) {
            this.labels = labels;
        }

        @Override
        public Tensor forward(Tensor... in) {
            Shape s = in[0].shape();
            int b = s.dim(0);
            int frames = s.dim(1);
            int c = s.dim(2);
            if (b != labels.length) {
                throw new IllegalArgumentException(labels.length + " label sequences for a batch of " + b);
            }
            float[] v = in[0].toArray();
            gradients = new double[b][][];
            double total = 0;
            for (int i = 0; i < b; i++) {
                double[][] logp = new double[frames][c];
                for (int f = 0; f < frames; f++) {
                    double max = Double.NEGATIVE_INFINITY;
                    for (int k = 0; k < c; k++) {
                        max = Math.max(max, v[(i * frames + f) * c + k]);
                    }
                    double sum = 0;
                    for (int k = 0; k < c; k++) {
                        sum += Math.exp(v[(i * frames + f) * c + k] - max);
                    }
                    double lse = max + Math.log(sum);
                    for (int k = 0; k < c; k++) {
                        logp[f][k] = v[(i * frames + f) * c + k] - lse;
                    }
                }
                double[] r = sequence(logp, labels[i], i);
                total += r[0];
            }
            return Tensor.of(Shape.of(), (float) (total / b)).to(in[0].device());
        }

        /** Forward-backward for one sequence; fills the gradient row and returns {@code {loss}}. */
        private double[] sequence(double[][] logp, int[] label, int index) {
            int frames = logp.length;
            int c = logp[0].length;
            int sLen = 2 * label.length + 1;   // blank, l1, blank, l2, ..., blank
            int[] ext = new int[sLen];
            for (int j = 0; j < label.length; j++) {
                ext[2 * j + 1] = label[j];
            }
            double ninf = Double.NEGATIVE_INFINITY;
            double[][] alpha = new double[frames][sLen];
            double[][] beta = new double[frames][sLen];
            for (double[] row : alpha) {
                Arrays.fill(row, ninf);
            }
            for (double[] row : beta) {
                Arrays.fill(row, ninf);
            }
            alpha[0][0] = logp[0][ext[0]];
            if (sLen > 1) {
                alpha[0][1] = logp[0][ext[1]];
            }
            for (int t = 1; t < frames; t++) {
                for (int s = 0; s < sLen; s++) {
                    double a = alpha[t - 1][s];
                    if (s > 0) {
                        a = logAdd(a, alpha[t - 1][s - 1]);
                    }
                    if (s > 1 && ext[s] != 0 && ext[s] != ext[s - 2]) {
                        a = logAdd(a, alpha[t - 1][s - 2]);
                    }
                    alpha[t][s] = a + logp[t][ext[s]];
                }
            }
            beta[frames - 1][sLen - 1] = logp[frames - 1][ext[sLen - 1]];
            if (sLen > 1) {
                beta[frames - 1][sLen - 2] = logp[frames - 1][ext[sLen - 2]];
            }
            for (int t = frames - 2; t >= 0; t--) {
                for (int s = 0; s < sLen; s++) {
                    double bb = beta[t + 1][s];
                    if (s + 1 < sLen) {
                        bb = logAdd(bb, beta[t + 1][s + 1]);
                    }
                    if (s + 2 < sLen && ext[s] != 0 && ext[s] != ext[s + 2]) {
                        bb = logAdd(bb, beta[t + 1][s + 2]);
                    }
                    beta[t][s] = bb + logp[t][ext[s]];
                }
            }
            double logLik = sLen > 1 ? logAdd(alpha[frames - 1][sLen - 1], alpha[frames - 1][sLen - 2])
                    : alpha[frames - 1][0];
            double[][] grad = new double[frames][c];
            for (int t = 0; t < frames; t++) {
                double[] occupancy = new double[c];
                Arrays.fill(occupancy, ninf);
                for (int s = 0; s < sLen; s++) {
                    // alpha and beta both include frame t's emission; remove one copy
                    occupancy[ext[s]] = logAdd(occupancy[ext[s]], alpha[t][s] + beta[t][s] - logp[t][ext[s]]);
                }
                for (int k = 0; k < c; k++) {
                    grad[t][k] = Math.exp(logp[t][k]) - (occupancy[k] == ninf ? 0 : Math.exp(occupancy[k] - logLik));
                }
            }
            gradients[index] = grad;
            return new double[] {-logLik};
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
            Shape s = in[0].shape();
            int b = s.dim(0);
            int frames = s.dim(1);
            int c = s.dim(2);
            float g = grad.item() / b;
            float[] d = new float[b * frames * c];
            for (int i = 0; i < b; i++) {
                for (int t = 0; t < frames; t++) {
                    for (int k = 0; k < c; k++) {
                        d[(i * frames + t) * c + k] = (float) (g * gradients[i][t][k]);
                    }
                }
            }
            return new Tensor[] {Tensor.of(s, d).to(in[0].device())};
        }

        private static double logAdd(double a, double b) {
            if (a == Double.NEGATIVE_INFINITY) {
                return b;
            }
            if (b == Double.NEGATIVE_INFINITY) {
                return a;
            }
            double m = Math.max(a, b);
            return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
        }
    }
}
