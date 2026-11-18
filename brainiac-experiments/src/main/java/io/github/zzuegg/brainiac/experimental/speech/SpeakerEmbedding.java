package io.github.zzuegg.brainiac.experimental.speech;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Speaker embeddings for verification (x-vectors, Snyder et al., ICASSP 2018) and their
 * training losses and scoring back-ends.
 *
 * <p>The network maps frame features {@code [batch, frames, bands]} to an embedding
 * {@code [batch, dim]}: time-delay (dilated 1-D) convolutions with dilations 1, 2, 3 and a
 * 1x1 layer, then <em>statistics pooling</em> (the mean and standard deviation of every channel
 * over time, which turns a variable-length utterance into a fixed vector) or <em>attentive
 * statistics pooling</em> (Okabe et al., 2018: a learned per-frame weight, softmaxed over time,
 * for a weighted mean and standard deviation), then a linear embedding layer.
 *
 * <p>Losses: {@link #aamSoftmax} (additive angular margin, ArcFace-style
 * {@code s cos(theta + m)} on the target class with the easy-margin fix; a local copy until
 * {@code experimental.metric.MarginSoftmax} lands) and {@link #ge2e} (Wan et al., ICASSP 2018):
 * on a batch of N speakers x M utterances each embedding should be near its own speaker's
 * centroid computed <em>without</em> it and far from other centroids, a softmax over
 * {@code w cos + b}. Scoring: cosine, and two-covariance PLDA ({@link Plda}). Metrics: equal
 * error rate ({@link #equalErrorRate}) and the normalised minimum detection cost
 * ({@link #minDcf}).
 *
 * <p>Papers: <a href="https://doi.org/10.1109/ICASSP.2018.8461375">doi:10.1109/ICASSP.2018.8461375</a>,
 * <a href="https://arxiv.org/abs/1710.10467">arXiv:1710.10467</a>,
 * <a href="https://arxiv.org/abs/1801.07698">arXiv:1801.07698</a>.
 */
public final class SpeakerEmbedding implements Layer {

    private static final int[] KERNELS = {5, 3, 3, 1};
    private static final int[] DILATIONS = {1, 2, 3, 1};

    private final int bands;
    private final int channels;
    private final int dim;
    private final boolean attentive;
    private final List<Var[]> tdnn = new ArrayList<>();
    private final Var attention;
    private final Var attentionBias;
    private final Var attentionOut;
    private final Var embed;
    private final Var embedBias;

    /**
     * Creates the network.
     *
     * @param bands input features per frame
     * @param channels TDNN channels
     * @param dim the embedding size
     * @param attentive attentive statistics pooling instead of plain
     * @param random the source of initial weights
     */
    public SpeakerEmbedding(int bands, int channels, int dim, boolean attentive, RandomGenerator random) {
        this.bands = bands;
        this.channels = channels;
        this.dim = dim;
        this.attentive = attentive;
        int in = bands;
        for (int l = 0; l < KERNELS.length; l++) {
            int k = KERNELS[l];
            tdnn.add(new Var[] {Var.param(Tensor.normal(Shape.of(channels, in, k), 0,
                    (float) Math.sqrt(2.0 / (in * k)), random)), Var.param(Tensor.zeros(Shape.of(channels)))});
            in = channels;
        }
        int hidden = Math.max(4, channels / 4);
        attention = Var.param(Tensor.normal(Shape.of(channels, hidden), 0, (float) Math.sqrt(1.0 / channels), random));
        attentionBias = Var.param(Tensor.zeros(Shape.of(hidden)));
        attentionOut = Var.param(Tensor.normal(Shape.of(hidden, 1), 0, (float) Math.sqrt(1.0 / hidden), random));
        embed = Var.param(Tensor.normal(Shape.of(2 * channels, dim), 0, (float) Math.sqrt(1.0 / (2 * channels)), random));
        embedBias = Var.param(Tensor.zeros(Shape.of(dim)));
    }

    /** @return the embedding size */
    public int dim() {
        return dim;
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        Var x = input.permute(0, 2, 1);
        for (int l = 0; l < tdnn.size(); l++) {
            int k = KERNELS[l];
            int d = DILATIONS[l];
            int pad = d * (k - 1) / 2;
            x = Functional.conv(x, tdnn.get(l)[0], tdnn.get(l)[1], new int[] {1}, new int[] {pad}, new int[] {pad},
                    new int[] {d}, 1).relu();
        }
        Var frames = x.permute(0, 2, 1); // [b, T, C]
        Var mean;
        Var std;
        if (attentive) {
            Var score = frames.matmul(attention).add(attentionBias).tanh().matmul(attentionOut); // [b, T, 1]
            Var w = score.sub(score.max(1, true).detach()).exp();
            w = w.div(w.sum(1, true));
            mean = frames.mul(w).sum(1, false);
            Var second = frames.square().mul(w).sum(1, false);
            std = second.sub(mean.square()).relu().add(1e-5f).sqrt();
        } else {
            mean = frames.mean(1, false);
            std = frames.sub(frames.mean(1, true)).square().mean(1, false).add(1e-5f).sqrt();
        }
        return Var.concat(1, mean, std).reshape(b, 2 * channels).matmul(embed).add(embedBias);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(dim);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Var[] l : tdnn) {
            p.addAll(Arrays.asList(l));
        }
        if (attentive) {
            p.addAll(List.of(attention, attentionBias, attentionOut));
        }
        p.add(embed);
        p.add(embedBias);
        return p;
    }

    /**
     * Embeds one utterance.
     *
     * @param features {@code [frames][bands]}
     * @return the embedding
     */
    public double[] embed(float[][] features) {
        Tensor t = SpeechFeatures.batch(List.<float[][]>of(features));
        return SpeechFeatures.toDouble(Var.noGrad(() -> forward(Var.of(t))).value().toArray());
    }

    @Override
    public String toString() {
        return "SpeakerEmbedding(" + bands + " bands, " + channels + " channels, " + dim + "d"
                + (attentive ? ", attentive pooling" : "") + ")";
    }

    // ---------------------------------------------------------------- losses

    private static Var l2normalise(Var x, int axis) {
        return x.div(x.square().sum(axis, true).add(1e-10f).sqrt());
    }

    /**
     * Additive angular margin softmax cross-entropy: logits {@code s cos(theta_j)}, with
     * {@code s cos(theta_y + m)} for the true class {@code y} (and the easy-margin fix
     * {@code cos(theta) - m sin(m)} where {@code theta + m} would pass pi).
     *
     * @param embeddings {@code [b, dim]}
     * @param classWeights {@code [dim, classes]}
     * @param labels {@code [b]} class ids
     * @param scale the scale {@code s} (about 30)
     * @param margin the margin {@code m} in radians (0 gives the normalised softmax)
     * @return the mean loss
     */
    public static Var aamSoftmax(Var embeddings, Var classWeights, int[] labels, float scale, float margin) {
        int b = embeddings.shape().dim(0);
        int classes = classWeights.shape().dim(1);
        Var cos = l2normalise(embeddings, 1).matmul(l2normalise(classWeights, 0)).clamp(-1 + 1e-6f, 1 - 1e-6f);
        float[] onehot = new float[b * classes];
        float[] ids = new float[b];
        for (int i = 0; i < b; i++) {
            onehot[i * classes + labels[i]] = 1;
            ids[i] = labels[i];
        }
        Tensor mask = Tensor.of(Shape.of(b, classes), onehot);
        Var sin = cos.square().neg().add(1f).sqrt();
        float cm = (float) Math.cos(margin);
        float sm = (float) Math.sin(margin);
        Var phi = cos.mul(cm).sub(sin.mul(sm));
        float th = (float) Math.cos(Math.PI - margin);
        Var fixedPhi = Var.where(cos.greater(th), phi, cos.sub((float) (Math.sin(Math.PI - margin) * margin)));
        Var logits = cos.add(fixedPhi.sub(cos).mul(mask)).mul(scale);
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(b), ids));
    }

    /**
     * The generalised end-to-end (softmax) loss on a batch ordered speaker by speaker.
     *
     * @param embeddings {@code [n * m, dim]}, rows {@code j * m .. (j + 1) * m - 1} from speaker j
     * @param speakers N
     * @param perSpeaker M (at least 2)
     * @param w the learned scale (kept positive by the caller, initialised to 10)
     * @param bias the learned offset (initialised to -5)
     * @return the mean loss
     */
    public static Var ge2e(Var embeddings, int speakers, int perSpeaker, Var w, Var bias) {
        int nm = speakers * perSpeaker;
        int d = embeddings.shape().dim(1);
        Var e = l2normalise(embeddings, 1);
        Var grouped = e.reshape(speakers, perSpeaker, d);
        Var sums = grouped.sum(1, false);                                  // [N, d]
        Var centroids = l2normalise(sums.mul(1f / perSpeaker), 1);
        Var sim = e.matmul(centroids.transpose());                         // [NM, N]
        float[] own = new float[nm];
        float[] mask = new float[nm * speakers];
        float[] ids = new float[nm];
        for (int i = 0; i < nm; i++) {
            own[i] = i / perSpeaker;
            ids[i] = i / perSpeaker;
            mask[i * speakers + i / perSpeaker] = 1;
        }
        Var ownSums = sums.gather(0, Tensor.of(Shape.of(nm), own));        // [NM, d]
        Var exclusive = l2normalise(ownSums.sub(e).mul(1f / (perSpeaker - 1)), 1);
        Var ownCos = e.mul(exclusive).sum(1, true);                        // [NM, 1]
        Tensor m = Tensor.of(Shape.of(nm, speakers), mask);
        Var mixed = sim.add(ownCos.sub(sim).mul(m));
        Var logits = mixed.mul(w).add(bias);
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(nm), ids));
    }

    // ---------------------------------------------------------------- scoring

    /**
     * Two-covariance PLDA: embeddings are {@code x = mu + y + e} with speaker variable
     * {@code y ~ N(0, B)} and residual {@code e ~ N(0, W)}; a trial is scored by the log
     * likelihood ratio of "same speaker" against "different speakers", computed in the basis
     * that whitens {@code W} and diagonalises {@code B}.
     */
    public static final class Plda {
        private final double[] mean;
        private final double[][] transform;
        private final double[] psi;

        private Plda(double[] mean, double[][] transform, double[] psi) {
            this.mean = mean;
            this.transform = transform;
            this.psi = psi;
        }

        /**
         * Estimates the model from labelled training embeddings.
         *
         * @param x the embeddings
         * @param speaker the speaker of each
         * @return the model
         */
        public static Plda fit(List<double[]> x, int[] speaker) {
            int d = x.get(0).length;
            int classes = Arrays.stream(speaker).max().orElse(0) + 1;
            double[] mu = new double[d];
            for (double[] v : x) {
                for (int i = 0; i < d; i++) {
                    mu[i] += v[i] / x.size();
                }
            }
            double[][] cm = new double[classes][d];
            int[] counts = new int[classes];
            for (int n = 0; n < x.size(); n++) {
                counts[speaker[n]]++;
                for (int i = 0; i < d; i++) {
                    cm[speaker[n]][i] += x.get(n)[i];
                }
            }
            double[][] within = new double[d][d];
            double[][] between = new double[d][d];
            int used = 0;
            for (int c = 0; c < classes; c++) {
                if (counts[c] == 0) {
                    continue;
                }
                used++;
                for (int i = 0; i < d; i++) {
                    cm[c][i] /= counts[c];
                }
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        between[i][j] += (cm[c][i] - mu[i]) * (cm[c][j] - mu[j]);
                    }
                }
            }
            for (int n = 0; n < x.size(); n++) {
                double[] c = cm[speaker[n]];
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        within[i][j] += (x.get(n)[i] - c[i]) * (x.get(n)[j] - c[j]);
                    }
                }
            }
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    within[i][j] /= x.size();
                    between[i][j] /= Math.max(1, used);
                }
                within[i][i] += 1e-6;
            }
            // whiten W
            Tensor[] ew = Linalg.eigh(matrix(within));
            double[] wv = ew[0].toDoubleArray();
            double[] wvec = ew[1].toDoubleArray();
            double[][] t1 = new double[d][d];
            for (int k = 0; k < d; k++) {
                double s = 1 / Math.sqrt(Math.max(wv[k], 1e-9));
                for (int i = 0; i < d; i++) {
                    t1[k][i] = wvec[i * d + k] * s;
                }
            }
            double[][] bw = mul(mul(t1, between), transpose(t1));
            Tensor[] eb = Linalg.eigh(matrix(bw));
            double[] psi = eb[0].toDoubleArray();
            double[] bvec = eb[1].toDoubleArray();
            double[][] u = new double[d][d];
            for (int k = 0; k < d; k++) {
                for (int i = 0; i < d; i++) {
                    u[k][i] = bvec[i * d + k];
                }
                psi[k] = Math.max(psi[k], 1e-6);
            }
            return new Plda(mu, mul(u, t1), psi);
        }

        private double[] project(double[] x) {
            int d = mean.length;
            double[] out = new double[d];
            for (int k = 0; k < d; k++) {
                double s = 0;
                for (int i = 0; i < d; i++) {
                    s += transform[k][i] * (x[i] - mean[i]);
                }
                out[k] = s;
            }
            return out;
        }

        /**
         * The log likelihood ratio of a trial.
         *
         * @param a one embedding
         * @param b another
         * @return same-speaker minus different-speaker log likelihood
         */
        public double score(double[] a, double[] b) {
            double[] u = project(a);
            double[] v = project(b);
            double llr = 0;
            for (int k = 0; k < psi.length; k++) {
                double p = psi[k];
                double sq = u[k] * u[k] + v[k] * v[k];
                llr += -0.5 * Math.log(2 * p + 1) + Math.log(p + 1)
                        - 0.5 * ((p + 1) * sq - 2 * p * u[k] * v[k]) / (2 * p + 1) + 0.5 * sq / (p + 1);
            }
            return llr;
        }

        private static Tensor matrix(double[][] m) {
            int d = m.length;
            float[] f = new float[d * d];
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    f[i * d + j] = (float) (0.5 * (m[i][j] + m[j][i]));
                }
            }
            return Tensor.of(Shape.of(d, d), f);
        }

        private static double[][] mul(double[][] a, double[][] b) {
            int n = a.length;
            int k = b.length;
            int m = b[0].length;
            double[][] c = new double[n][m];
            for (int i = 0; i < n; i++) {
                for (int l = 0; l < k; l++) {
                    double v = a[i][l];
                    for (int j = 0; j < m; j++) {
                        c[i][j] += v * b[l][j];
                    }
                }
            }
            return c;
        }

        private static double[][] transpose(double[][] a) {
            double[][] t = new double[a[0].length][a.length];
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < a[0].length; j++) {
                    t[j][i] = a[i][j];
                }
            }
            return t;
        }
    }

    // ---------------------------------------------------------------- metrics

    /**
     * The equal error rate: the miss rate at the threshold where it equals the false-alarm
     * rate (interpolated between the two neighbouring thresholds).
     *
     * @param scores trial scores (higher means same speaker)
     * @param target whether each trial is a same-speaker trial
     * @return the EER in [0, 1]
     */
    public static double equalErrorRate(double[] scores, boolean[] target) {
        double[][] curve = curve(scores, target);
        double best = 1;
        for (int i = 0; i < curve.length; i++) {
            double miss = curve[i][0];
            double fa = curve[i][1];
            if (miss >= fa) {
                if (i == 0) {
                    return (miss + fa) / 2;
                }
                double pm = curve[i - 1][0];
                double pf = curve[i - 1][1];
                double t = (pf - pm) / ((pf - pm) - (fa - miss) + 1e-12);
                return pm + t * (miss - pm);
            }
            best = Math.min(best, Math.max(miss, fa));
        }
        return best;
    }

    /**
     * The normalised minimum detection cost {@code min_t (P_miss(t) p + P_fa(t) (1 - p)) /
     * min(p, 1 - p)} with unit costs.
     *
     * @param scores trial scores
     * @param target whether each trial is a same-speaker trial
     * @param pTarget the prior of a target trial (such as 0.05)
     * @return the cost
     */
    public static double minDcf(double[] scores, boolean[] target, double pTarget) {
        double best = Double.MAX_VALUE;
        for (double[] c : curve(scores, target)) {
            best = Math.min(best, c[0] * pTarget + c[1] * (1 - pTarget));
        }
        return best / Math.min(pTarget, 1 - pTarget);
    }

    /** Per threshold (ascending, starting below every score) {miss, false alarm}. */
    private static double[][] curve(double[] scores, boolean[] target) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        int pos = 0;
        for (boolean t : target) {
            if (t) {
                pos++;
            }
        }
        int neg = scores.length - pos;
        double[][] out = new double[scores.length + 1][2];
        int missed = 0;
        int rejectedNeg = 0;
        out[0] = new double[] {0, 1};
        for (int i = 0; i < order.length; i++) {
            if (target[order[i]]) {
                missed++;
            } else {
                rejectedNeg++;
            }
            out[i + 1] = new double[] {missed / (double) Math.max(1, pos), (neg - rejectedNeg) / (double) Math.max(1, neg)};
        }
        return out;
    }
}
