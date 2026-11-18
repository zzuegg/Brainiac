package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * The RNN-Transducer loss and its decoders. CTC assumes the labels are conditionally
 * independent given the audio; the transducer adds a prediction network over the labels
 * emitted so far and a joint network {@code z(t, u) = J(enc_t, pred_u)}, so every output
 * distribution sees the label context. The loss is {@code -log P(y | x)}, the sum over all
 * monotone alignments on the {@code T x (U + 1)} lattice, where each node either emits
 * the blank (advance one frame) or the next label (stay on the frame). It is computed by
 * a forward ({@code alpha}) and backward ({@code beta}) recursion in log space, and the
 * gradient with respect to each log-probability is
 * {@code -exp(alpha(t,u) + logp + beta(next) - log P)}, written directly (a custom op).
 *
 * <p>{@link Joint} is a small transducer head: a stateless prediction network (an
 * embedding of the previous label, which suffices for a first-order label context) and an
 * additive joint {@code tanh(enc W_e + pred W_p) W_o}; it gives the lattice logits, greedy
 * decoding and a frame-synchronous beam search (at most one label per frame, hypotheses
 * with equal labels merged). Class 0 is the blank.
 *
 * <p>Paper: Graves, "Sequence Transduction with Recurrent Neural Networks" (2012),
 * <a href="https://arxiv.org/abs/1211.3711">arXiv:1211.3711</a>; stateless prediction
 * network: Ghodsi, Liu, Moreno, "Rnn-Transducer with Stateless Prediction Network" (ICASSP 2020).
 */
public final class TransducerLoss {

    private TransducerLoss() {
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

    /**
     * The mean transducer loss over a batch.
     *
     * @param logProbs log-probabilities {@code [batch, frames, maxLabels + 1, classes]}
     * @param labels per example its labels {@code 1..classes-1}, at most {@code maxLabels}
     * @return the mean of {@code -log P(labels | input)}
     */
    public static Var loss(Var logProbs, int[][] labels) {
        int b = logProbs.shape().dim(0);
        int t = logProbs.shape().dim(1);
        int u1 = logProbs.shape().dim(2);
        int v = logProbs.shape().dim(3);
        double[][][] alphas = new double[b][][];
        double[][][] betas = new double[b][][];
        double[] logP = new double[b];
        return Var.custom(in -> {
            float[] lp = in[0].toArray();
            double total = 0;
            for (int i = 0; i < b; i++) {
                int[] y = labels[i];
                int u = y.length;
                double[][] a = new double[t][u + 1];
                double[][] be = new double[t][u + 1];
                for (int ti = 0; ti < t; ti++) {
                    for (int ui = 0; ui <= u; ui++) {
                        double val;
                        if (ti == 0 && ui == 0) {
                            val = 0;
                        } else {
                            val = Double.NEGATIVE_INFINITY;
                            if (ti > 0) {
                                val = logAdd(val, a[ti - 1][ui] + lp[idx(i, ti - 1, ui, 0, t, u1, v)]);
                            }
                            if (ui > 0) {
                                val = logAdd(val, a[ti][ui - 1] + lp[idx(i, ti, ui - 1, y[ui - 1], t, u1, v)]);
                            }
                        }
                        a[ti][ui] = val;
                    }
                }
                for (int ti = t - 1; ti >= 0; ti--) {
                    for (int ui = u; ui >= 0; ui--) {
                        double val;
                        if (ti == t - 1 && ui == u) {
                            val = lp[idx(i, ti, ui, 0, t, u1, v)];
                        } else {
                            val = Double.NEGATIVE_INFINITY;
                            if (ti < t - 1) {
                                val = logAdd(val, be[ti + 1][ui] + lp[idx(i, ti, ui, 0, t, u1, v)]);
                            }
                            if (ui < u) {
                                val = logAdd(val, be[ti][ui + 1] + lp[idx(i, ti, ui, y[ui], t, u1, v)]);
                            }
                        }
                        be[ti][ui] = val;
                    }
                }
                alphas[i] = a;
                betas[i] = be;
                logP[i] = be[0][0];
                total -= logP[i];
            }
            return Tensor.full(Shape.of(), (float) (total / b));
        }, (in, out, grad) -> {
            float[] lp = in[0].toArray();
            float scale = grad.item() / b;
            float[] g = new float[lp.length];
            for (int i = 0; i < b; i++) {
                int[] y = labels[i];
                int u = y.length;
                double[][] a = alphas[i];
                double[][] be = betas[i];
                for (int ti = 0; ti < t; ti++) {
                    for (int ui = 0; ui <= u; ui++) {
                        int blank = idx(i, ti, ui, 0, t, u1, v);
                        double next = ti == t - 1 ? (ui == u ? 0 : Double.NEGATIVE_INFINITY) : be[ti + 1][ui];
                        if (next != Double.NEGATIVE_INFINITY) {
                            g[blank] -= (float) (scale * Math.exp(a[ti][ui] + lp[blank] + next - logP[i]));
                        }
                        if (ui < u) {
                            int emit = idx(i, ti, ui, y[ui], t, u1, v);
                            g[emit] -= (float) (scale * Math.exp(a[ti][ui] + lp[emit] + be[ti][ui + 1] - logP[i]));
                        }
                    }
                }
            }
            return new Tensor[] {Tensor.of(in[0].shape(), g)};
        }, logProbs);
    }

    private static int idx(int b, int t, int u, int k, int frames, int u1, int v) {
        return ((b * frames + t) * u1 + u) * v + k;
    }

    /** A stateless prediction network and an additive joint network. */
    public static final class Joint {
        private final int classes;
        private final Var embedding;
        private final Var encProj;
        private final Var predProj;
        private final Var bias;
        private final Var out;

        /**
         * Creates the head.
         *
         * @param encoderWidth the encoder's output width
         * @param joint the joint network's width
         * @param classes output classes, the blank included
         * @param random the random source
         */
        public Joint(int encoderWidth, int joint, int classes, SplittableRandom random) {
            this.classes = classes;
            embedding = Var.param(Tensor.normal(Shape.of(classes, joint), 0, 0.3f, random));
            encProj = MixerOps.weight(random, encoderWidth, joint);
            predProj = MixerOps.weight(random, joint, joint);
            bias = MixerOps.filled(0f, joint);
            out = MixerOps.weight(random, joint, classes);
        }

        /** Prediction-network outputs for given previous labels {@code [b][n]} (0 = start). */
        private Var predict(int[][] previous) {
            int b = previous.length;
            int n = previous[0].length;
            float[] ids = new float[b * n];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < n; j++) {
                    ids[i * n + j] = previous[i][j];
                }
            }
            return embedding.gather(0, Tensor.of(Shape.of(b, n), ids)).matmul(predProj);
        }

        /**
         * Lattice log-probabilities for training.
         *
         * @param encoded encoder outputs {@code [b, frames, width]}
         * @param labels per example its labels, all of the same length {@code U}
         * @return {@code [b, frames, U + 1, classes]} log-probabilities
         */
        public Var logProbs(Var encoded, int[][] labels) {
            int b = encoded.shape().dim(0);
            int t = encoded.shape().dim(1);
            int u = labels[0].length;
            int[][] prev = new int[b][u + 1];
            for (int i = 0; i < b; i++) {
                System.arraycopy(labels[i], 0, prev[i], 1, u);
            }
            int j = bias.shape().dim(0);
            Var e = encoded.matmul(encProj).reshape(b, t, 1, j);
            Var p = predict(prev).reshape(b, 1, u + 1, j);
            Var z = e.add(p).add(bias).tanh().matmul(out);
            return z.sub(z.logsumexp(3, true));
        }

        private float[] step(Tensor encFrame, int previous) {
            Var e = Var.of(encFrame).matmul(encProj);
            Var p = predict(new int[][] {{previous}}).reshape(1, bias.shape().dim(0));
            Var z = e.add(p).add(bias).tanh().matmul(out);
            return z.sub(z.logsumexp(1, true)).value().toArray();
        }

        /**
         * Greedy decoding, at most {@code 3} labels per frame.
         *
         * @param encoded one utterance's encoder outputs {@code [frames, width]}
         * @param emissionFrames receives the frame of each emitted label (may be null)
         * @return the labels
         */
        public int[] greedy(Tensor encoded, List<Integer> emissionFrames) {
            return Var.noGrad(() -> {
                int t = encoded.shape().dim(0);
                List<Integer> outLabels = new ArrayList<>();
                int prev = 0;
                for (int ti = 0; ti < t; ti++) {
                    Tensor frame = encoded.slice(0, ti, ti + 1);
                    for (int s = 0; s < 3; s++) {
                        float[] lp = step(frame, prev);
                        int best = 0;
                        for (int k = 1; k < classes; k++) {
                            if (lp[k] > lp[best]) {
                                best = k;
                            }
                        }
                        if (best == 0) {
                            break;
                        }
                        outLabels.add(best);
                        if (emissionFrames != null) {
                            emissionFrames.add(ti);
                        }
                        prev = best;
                    }
                }
                return outLabels.stream().mapToInt(Integer::intValue).toArray();
            });
        }

        private record Hyp(List<Integer> labels, double score) {
        }

        /**
         * Frame-synchronous beam search with at most one label per frame; hypotheses with
         * the same labels are merged by adding their probabilities.
         *
         * @param encoded one utterance's encoder outputs {@code [frames, width]}
         * @param width the beam width
         * @return the best labels
         */
        public int[] beam(Tensor encoded, int width) {
            return Var.noGrad(() -> {
                int t = encoded.shape().dim(0);
                List<Hyp> beam = List.of(new Hyp(List.of(), 0));
                for (int ti = 0; ti < t; ti++) {
                    Tensor frame = encoded.slice(0, ti, ti + 1);
                    Map<List<Integer>, Double> next = new HashMap<>();
                    for (Hyp h : beam) {
                        int prev = h.labels().isEmpty() ? 0 : h.labels().get(h.labels().size() - 1);
                        float[] lp = step(frame, prev);
                        next.merge(h.labels(), h.score() + lp[0], TransducerLoss::logAdd);
                        for (int k = 1; k < classes; k++) {
                            List<Integer> l = new ArrayList<>(h.labels());
                            l.add(k);
                            next.merge(l, h.score() + lp[k], TransducerLoss::logAdd);
                        }
                    }
                    List<Hyp> all = new ArrayList<>();
                    next.forEach((k, s) -> all.add(new Hyp(k, s)));
                    all.sort(Comparator.comparingDouble(Hyp::score).reversed());
                    beam = all.subList(0, Math.min(width, all.size()));
                }
                return beam.get(0).labels().stream().mapToInt(Integer::intValue).toArray();
            });
        }

        /** @return the head's weights */
        public List<Var> parameters() {
            return List.of(embedding, encProj, predProj, bias, out);
        }
    }
}
