package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Connectionist temporal classification (Graves et al., 2006): trains a model that emits a
 * distribution over labels plus a blank at each time step to output label sequences shorter
 * than its input, without aligning them, as speech and handwriting recognizers do. Decodes
 * greedily or with a prefix beam search.
 */
public final class Ctc {

    private Ctc() {
    }

    /**
     * The CTC loss: the negative log-probability of each target summed over all its
     * alignments, averaged over the batch. Computed by the forward-backward recursion on the
     * host; the gradient is exact.
     *
     * @param logProbs log-softmax outputs {@code [batch, time, classes]}
     * @param targets per example its labels, each in {@code [0, classes)} and not the blank
     * @param inputLengths per example how many time steps count; {@code null} for all
     * @param blank the blank class, usually 0
     * @return a scalar
     */
    public static Var loss(Var logProbs, int[][] targets, int[] inputLengths, int blank) {
        Shape s = logProbs.shape();
        if (s.rank() != 3 || s.dim(0) != targets.length) {
            throw new IllegalArgumentException("needs [batch, time, classes] and one target per example, got " + s
                    + " and " + targets.length);
        }
        int batch = s.dim(0);
        int time = s.dim(1);
        int classes = s.dim(2);
        int[] lengths = inputLengths != null ? inputLengths : filled(batch, time);
        Op op = new Op() {
            private double[] grads;

            @Override
            public Tensor forward(Tensor... in) {
                double[] lp = in[0].toDoubleArray();
                grads = new double[lp.length];
                double total = 0;
                for (int b = 0; b < batch; b++) {
                    total += example(lp, b * time * classes, lengths[b], classes, targets[b], blank, grads);
                }
                for (int i = 0; i < grads.length; i++) {
                    grads[i] /= batch;
                }
                return Tensor.full(Shape.of(), (float) (total / batch), in[0].dtype(), in[0].device());
            }

            @Override
            public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
                Tensor g = Tensor.of(in[0].shape(), grads, in[0].dtype(), in[0].device());
                return new Tensor[] {g.mul(grad)};
            }

            @Override
            public String name() {
                return "ctc";
            }
        };
        return Var.apply(op, logProbs);
    }

    private static int[] filled(int n, int v) {
        int[] a = new int[n];
        java.util.Arrays.fill(a, v);
        return a;
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

    /** One example's loss; adds its gradient with respect to the log-probabilities into {@code grads}. */
    private static double example(double[] lp, int base, int time, int classes, int[] target, int blank,
            double[] grads) {
        int l = target.length;
        int n = 2 * l + 1;
        int[] ext = new int[n];
        for (int i = 0; i < n; i++) {
            ext[i] = i % 2 == 0 ? blank : target[i / 2];
            if (ext[i] < 0 || ext[i] >= classes) {
                throw new IllegalArgumentException("label " + ext[i] + " out of range [0, " + classes + ")");
            }
        }
        double ninf = Double.NEGATIVE_INFINITY;
        double[][] alpha = new double[time][n];
        double[][] beta = new double[time][n];
        for (double[] row : alpha) {
            java.util.Arrays.fill(row, ninf);
        }
        for (double[] row : beta) {
            java.util.Arrays.fill(row, ninf);
        }
        alpha[0][0] = lp[base + ext[0]];
        if (n > 1) {
            alpha[0][1] = lp[base + ext[1]];
        }
        for (int t = 1; t < time; t++) {
            for (int s = 0; s < n; s++) {
                double a = alpha[t - 1][s];
                if (s > 0) {
                    a = logAdd(a, alpha[t - 1][s - 1]);
                }
                if (s > 1 && ext[s] != blank && ext[s] != ext[s - 2]) {
                    a = logAdd(a, alpha[t - 1][s - 2]);
                }
                alpha[t][s] = a + lp[base + t * classes + ext[s]];
            }
        }
        beta[time - 1][n - 1] = lp[base + (time - 1) * classes + ext[n - 1]];
        if (n > 1) {
            beta[time - 1][n - 2] = lp[base + (time - 1) * classes + ext[n - 2]];
        }
        for (int t = time - 2; t >= 0; t--) {
            for (int s = 0; s < n; s++) {
                double v = beta[t + 1][s];
                if (s + 1 < n) {
                    v = logAdd(v, beta[t + 1][s + 1]);
                }
                if (s + 2 < n && ext[s] != blank && ext[s] != ext[s + 2]) {
                    v = logAdd(v, beta[t + 1][s + 2]);
                }
                beta[t][s] = v + lp[base + t * classes + ext[s]];
            }
        }
        double logP = n > 1 ? logAdd(alpha[time - 1][n - 1], alpha[time - 1][n - 2]) : alpha[time - 1][n - 1];
        if (logP == ninf) {
            return Double.POSITIVE_INFINITY;   // the target does not fit in the input
        }
        // d(-log p)/d lp[t][c] = -exp(logsum over s with ext[s] = c of alpha + beta - lp - log p)
        for (int t = 0; t < time; t++) {
            Map<Integer, Double> sums = new HashMap<>();
            for (int s = 0; s < n; s++) {
                double ab = alpha[t][s] + beta[t][s];
                if (ab != ninf) {
                    sums.merge(ext[s], ab, Ctc::logAdd);
                }
            }
            for (Map.Entry<Integer, Double> e : sums.entrySet()) {
                int c = e.getKey();
                grads[base + t * classes + c] -= Math.exp(e.getValue() - lp[base + t * classes + c] - logP);
            }
        }
        return -logP;
    }

    /**
     * Greedy decoding: the most likely class at each step, repeats merged, blanks dropped.
     *
     * @param logProbs {@code [batch, time, classes]}
     * @param blank the blank class
     * @return per example the labels
     */
    public static int[][] greedy(Tensor logProbs, int blank) {
        int batch = logProbs.shape().dim(0);
        int time = logProbs.shape().dim(1);
        float[] best = logProbs.argmax(-1).toArray();
        int[][] out = new int[batch][];
        for (int b = 0; b < batch; b++) {
            List<Integer> labels = new ArrayList<>();
            int prev = -1;
            for (int t = 0; t < time; t++) {
                int c = (int) best[b * time + t];
                if (c != prev && c != blank) {
                    labels.add(c);
                }
                prev = c;
            }
            out[b] = labels.stream().mapToInt(Integer::intValue).toArray();
        }
        return out;
    }

    /**
     * Prefix beam search: keeps the {@code width} most likely label prefixes, summing the
     * probability of every alignment of each, so it finds sequences greedy decoding misses.
     *
     * @param logProbs one example's {@code [time, classes]}
     * @param width the beam width
     * @param blank the blank class
     * @return the most likely labels
     */
    public static int[] beamSearch(Tensor logProbs, int width, int blank) {
        int time = logProbs.shape().dim(0);
        int classes = logProbs.shape().dim(1);
        double[] lp = logProbs.toDoubleArray();
        double ninf = Double.NEGATIVE_INFINITY;
        // prefix -> {log p ending in blank, log p ending in its last label}
        Map<List<Integer>, double[]> beams = new HashMap<>();
        beams.put(List.of(), new double[] {0, ninf});
        for (int t = 0; t < time; t++) {
            Map<List<Integer>, double[]> next = new HashMap<>();
            for (Map.Entry<List<Integer>, double[]> e : beams.entrySet()) {
                List<Integer> prefix = e.getKey();
                double pb = e.getValue()[0];
                double pnb = e.getValue()[1];
                double total = logAdd(pb, pnb);
                int last = prefix.isEmpty() ? -1 : prefix.getLast();
                for (int c = 0; c < classes; c++) {
                    double p = lp[t * classes + c];
                    if (c == blank) {
                        double[] v = next.computeIfAbsent(prefix, k -> new double[] {ninf, ninf});
                        v[0] = logAdd(v[0], total + p);
                        continue;
                    }
                    List<Integer> extended = new ArrayList<>(prefix);
                    extended.add(c);
                    double[] v = next.computeIfAbsent(List.copyOf(extended), k -> new double[] {ninf, ninf});
                    if (c == last) {
                        v[1] = logAdd(v[1], pb + p);   // a repeat needs a blank between
                        double[] same = next.computeIfAbsent(prefix, k -> new double[] {ninf, ninf});
                        same[1] = logAdd(same[1], pnb + p);
                    } else {
                        v[1] = logAdd(v[1], total + p);
                    }
                }
            }
            Map<List<Integer>, double[]> kept = new HashMap<>();
            next.entrySet().stream()
                    .sorted(Comparator.comparingDouble((Map.Entry<List<Integer>, double[]> x) ->
                            -logAdd(x.getValue()[0], x.getValue()[1])))
                    .limit(width).forEach(x -> kept.put(x.getKey(), x.getValue()));
            beams = kept;
        }
        return beams.entrySet().stream()
                .max(Comparator.comparingDouble(x -> logAdd(x.getValue()[0], x.getValue()[1])))
                .orElseThrow().getKey().stream().mapToInt(Integer::intValue).toArray();
    }
}
