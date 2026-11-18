package io.github.zzuegg.brainiac.experimental.music;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.HiddenMarkovModel;
import io.github.zzuegg.brainiac.experimental.nn.LinearChainCrf;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Major/minor chord recognition over 25 labels (12 major, 12 minor, no chord; label
 * {@code root + 12 minor}, 24 for no chord).
 *
 * <ul>
 *   <li>{@link #template}: cosine similarity of each chroma frame to binary triad templates,
 *       no chord below an energy threshold, decided frame by frame or smoothed by an HMM
 *       whose transitions favour staying on the same chord (the templates' scaled
 *       similarities serve as emission log-scores);
 *   <li>{@link GaussianHmm}: one diagonal Gaussian per chord quality fitted to
 *       root-rotated training chroma (so the 12 roots share parameters, Sheh and Ellis), a
 *       no-chord Gaussian, a self-transition probability estimated from the training labels,
 *       and Viterbi decoding through {@link HiddenMarkovModel#viterbi(double[], double[][], double[][])};
 *   <li>{@link CnnCrf}: a small convolution over time on log-frequency spectra (no hand-made
 *       chroma) giving per-frame emission scores, trained end to end with a
 *       {@link LinearChainCrf} likelihood and decoded by its Viterbi.
 * </ul>
 *
 * <p>Paper: Sheh, Ellis, "Chord Segmentation and Recognition using EM-Trained Hidden Markov
 * Models", ISMIR 2003, <a href="https://doi.org/10.7916/D8V12P4J">doi:10.7916/D8V12P4J</a>.
 */
public final class ChordRecognizer {

    /** The number of labels. */
    public static final int LABELS = 25;
    /** The no-chord label. */
    public static final int NO_CHORD = 24;

    private ChordRecognizer() {
    }

    /**
     * The binary template of a label.
     *
     * @param label a chord label below 24
     * @return its 12-vector, unit norm
     */
    public static float[] templateOf(int label) {
        int root = label % 12;
        boolean minor = label >= 12;
        float[] t = new float[12];
        t[root] = 1;
        t[(root + (minor ? 3 : 4)) % 12] = 1;
        t[(root + 7) % 12] = 1;
        for (int c = 0; c < 12; c++) {
            t[c] /= (float) Math.sqrt(3);
        }
        return t;
    }

    private static double cosine(float[] a, float[] b) {
        double d = 0;
        double na = 1e-9;
        for (int i = 0; i < 12; i++) {
            d += a[i] * b[i];
            na += a[i] * a[i];
        }
        return d / Math.sqrt(na);
    }

    /**
     * Template matching.
     *
     * @param chroma per-frame chroma
     * @param energy per-frame energy
     * @param silence frames with energy below this get no chord
     * @param stay the HMM self-transition probability, or {@code 0} for frame-wise decisions
     * @return per-frame labels
     */
    public static int[] template(float[][] chroma, float[] energy, double silence, double stay) {
        int n = chroma.length;
        double[][] scores = new double[n][LABELS];
        for (int t = 0; t < n; t++) {
            boolean quiet = energy[t] < silence;
            for (int l = 0; l < 24; l++) {
                scores[t][l] = quiet ? -5 : 10 * cosine(chroma[t], templateOf(l));
            }
            scores[t][NO_CHORD] = quiet ? 10 : 10 * 0.4;
        }
        if (stay <= 0) {
            int[] out = new int[n];
            for (int t = 0; t < n; t++) {
                out[t] = argmax(scores[t]);
            }
            return out;
        }
        return HiddenMarkovModel.viterbi(uniformStart(), transitions(stay), scores);
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[b]) {
                b = i;
            }
        }
        return b;
    }

    private static double[] uniformStart() {
        double[] s = new double[LABELS];
        java.util.Arrays.fill(s, -Math.log(LABELS));
        return s;
    }

    private static double[][] transitions(double stay) {
        double[][] t = new double[LABELS][LABELS];
        for (int i = 0; i < LABELS; i++) {
            for (int j = 0; j < LABELS; j++) {
                t[i][j] = Math.log(i == j ? stay : (1 - stay) / (LABELS - 1));
            }
        }
        return t;
    }

    /** Gaussian emissions with root-tied parameters and HMM smoothing. */
    public static final class GaussianHmm {

        private final double[][] mean = new double[3][13];
        private final double[][] var = new double[3][13];
        private double stay;

        private GaussianHmm() {
        }

        private static double[] feature(float[] chroma, float energy, int root) {
            double[] f = new double[13];
            float[] r = Chroma.rotate(chroma, root);
            for (int c = 0; c < 12; c++) {
                f[c] = r[c];
            }
            f[12] = Math.log(1e-3 + energy);
            return f;
        }

        /**
         * Fits the model to labelled sequences.
         *
         * @param chroma per sequence its chroma frames
         * @param energy per sequence its frame energies
         * @param labels per sequence its frame labels
         * @return the model
         */
        public static GaussianHmm fit(List<float[][]> chroma, List<float[]> energy, List<int[]> labels) {
            GaussianHmm m = new GaussianHmm();
            double[] count = new double[3];
            double same = 0;
            double pairs = 0;
            for (int pass = 0; pass < 2; pass++) {
                for (int s = 0; s < chroma.size(); s++) {
                    int[] y = labels.get(s);
                    for (int t = 0; t < y.length; t++) {
                        int q = y[t] == NO_CHORD ? 2 : y[t] / 12;
                        double[] f = feature(chroma.get(s)[t], energy.get(s)[t], y[t] == NO_CHORD ? 0 : y[t] % 12);
                        if (pass == 0) {
                            count[q]++;
                            for (int d = 0; d < 13; d++) {
                                m.mean[q][d] += f[d];
                            }
                            if (t > 0) {
                                pairs++;
                                same += y[t] == y[t - 1] ? 1 : 0;
                            }
                        } else {
                            for (int d = 0; d < 13; d++) {
                                m.var[q][d] += Math.pow(f[d] - m.mean[q][d], 2);
                            }
                        }
                    }
                }
                for (int q = 0; q < 3; q++) {
                    for (int d = 0; d < 13; d++) {
                        if (pass == 0) {
                            m.mean[q][d] /= Math.max(1, count[q]);
                        } else {
                            m.var[q][d] = m.var[q][d] / Math.max(1, count[q]) + 1e-3;
                        }
                    }
                }
            }
            m.stay = Math.min(0.995, Math.max(0.5, same / Math.max(1, pairs)));
            return m;
        }

        private double logGauss(int q, double[] f) {
            double s = 0;
            for (int d = 0; d < 13; d++) {
                s -= 0.5 * (Math.pow(f[d] - mean[q][d], 2) / var[q][d] + Math.log(2 * Math.PI * var[q][d]));
            }
            return s;
        }

        /**
         * Decodes one sequence.
         *
         * @param chroma its chroma frames
         * @param energy its frame energies
         * @param smooth whether to Viterbi-decode (else frame-wise maximum likelihood)
         * @return per-frame labels
         */
        public int[] decode(float[][] chroma, float[] energy, boolean smooth) {
            double[][] e = new double[chroma.length][LABELS];
            for (int t = 0; t < chroma.length; t++) {
                for (int l = 0; l < 24; l++) {
                    e[t][l] = logGauss(l / 12, feature(chroma[t], energy[t], l % 12));
                }
                e[t][NO_CHORD] = logGauss(2, feature(chroma[t], energy[t], 0));
            }
            if (!smooth) {
                int[] out = new int[e.length];
                for (int t = 0; t < e.length; t++) {
                    out[t] = argmax(e[t]);
                }
                return out;
            }
            return HiddenMarkovModel.viterbi(uniformStart(), transitions(stay), e);
        }

        /** @return the estimated self-transition probability */
        public double stay() {
            return stay;
        }
    }

    /** A convolutional emission model with a linear-chain CRF on top. */
    public static final class CnnCrf {

        private final int features;
        private final Var w1;
        private final Var b1;
        private final Var w2;
        private final Var b2;
        private final LinearChainCrf crf;
        private final List<Var> params;

        /**
         * Creates an untrained model.
         *
         * @param features input features per frame
         * @param channels hidden channels
         * @param seed the seed
         */
        public CnnCrf(int features, int channels, long seed) {
            this.features = features;
            SplittableRandom r = new SplittableRandom(seed);
            w1 = Var.param(Tensor.normal(Shape.of(channels, features, 5), 0, (float) Math.sqrt(2.0 / (5 * features)),
                    r));
            b1 = Var.param(Tensor.zeros(Shape.of(channels)));
            w2 = Var.param(Tensor.normal(Shape.of(LABELS, channels, 3), 0, (float) Math.sqrt(1.0 / (3 * channels)),
                    r));
            b2 = Var.param(Tensor.zeros(Shape.of(LABELS)));
            crf = new LinearChainCrf(LABELS, r);
            List<Var> p = new ArrayList<>(List.of(w1, b1, w2, b2));
            p.addAll(crf.parameters());
            params = p;
        }

        /**
         * Emission scores.
         *
         * @param x {@code [batch, frames, features]}
         * @return {@code [batch, frames, LABELS]}
         */
        public Var emissions(Tensor x) {
            Var h = Var.of(x).permute(0, 2, 1);
            h = Functional.conv(h, w1, b1, new int[] {1}, new int[] {2}, new int[] {2}, new int[] {1}, 1).relu();
            h = Functional.conv(h, w2, b2, new int[] {1}, new int[] {1}, new int[] {1}, new int[] {1}, 1);
            return h.permute(0, 2, 1);
        }

        private static Tensor stack(List<float[][]> xs, int[] rows, int f) {
            int t = xs.get(rows[0]).length;
            float[] v = new float[rows.length * t * f];
            for (int i = 0; i < rows.length; i++) {
                float[][] x = xs.get(rows[i]);
                for (int j = 0; j < t; j++) {
                    System.arraycopy(x[j], 0, v, (i * t + j) * f, f);
                }
            }
            return Tensor.of(Shape.of(rows.length, t, f), v);
        }

        /**
         * Trains on labelled sequences of one length.
         *
         * @param x per sequence its feature frames
         * @param labels per sequence its labels
         * @param steps steps
         * @param batch sequences per step
         * @param lr learning rate
         * @param seed the seed
         * @return the last loss
         */
        public float train(List<float[][]> x, List<int[]> labels, int steps, int batch, float lr, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            Optimizer opt = Optimizer.adam(lr);
            int t = x.get(0).length;
            float last = 0;
            for (int s = 0; s < steps; s++) {
                int[] rows = new int[batch];
                float[] gold = new float[batch * t];
                for (int i = 0; i < batch; i++) {
                    rows[i] = r.nextInt(x.size());
                    for (int j = 0; j < t; j++) {
                        gold[i * t + j] = labels.get(rows[i])[j];
                    }
                }
                params.forEach(Var::zeroGrad);
                Var loss = crf.negativeLogLikelihood(emissions(stack(x, rows, features)),
                        Tensor.of(Shape.of(batch, t), gold));
                last = loss.value().item();
                loss.backward();
                opt.step(params);
            }
            params.forEach(Var::zeroGrad);
            return last;
        }

        /**
         * Decodes one sequence.
         *
         * @param x its feature frames
         * @return per-frame labels
         */
        public int[] decode(float[][] x) {
            Tensor e = Var.noGrad(() -> emissions(stack(java.util.Collections.singletonList(x), new int[] {0}, features))).value();
            return crf.decode(e)[0];
        }
    }
}
