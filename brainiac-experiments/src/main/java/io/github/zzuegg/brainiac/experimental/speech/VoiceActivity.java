package io.github.zzuegg.brainiac.experimental.speech;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.HiddenMarkovModel;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Voice activity detection: which frames contain speech.
 *
 * <ul>
 *   <li>{@link #energy}: frame log energy above an adaptive noise floor (the 10th percentile
 *       of the utterance's frame energies) plus a margin, with a zero-crossing-rate veto for
 *       noise-like frames ({@link #zeroCrossings});</li>
 *   <li>{@link #sohn} (Sohn, Kim and Sung, IEEE SPL 1999): speech and noise spectra modelled as
 *       complex Gaussians; per frequency the likelihood ratio
 *       {@code log L_k = gamma_k xi_k / (1 + xi_k) - log(1 + xi_k)} with the a-posteriori SNR
 *       {@code gamma_k} and the decision-directed a-priori SNR {@code xi_k}, averaged over
 *       frequencies; the noise spectrum starts from the quietest frames and is updated in
 *       frames judged noise;</li>
 *   <li>{@link #hmmHangover}: a two-state hidden Markov model with Gaussian emissions on the
 *       frame score ({@link HiddenMarkovModel}, fitted to the utterance by Baum-Welch) whose
 *       sticky transitions provide the hangover; the speech posterior is the output;</li>
 *   <li>{@link Crnn} (after Hughes and Mierle, ICASSP 2013): a 1-D convolution over log-mel
 *       frames, a GRU and a per-frame logistic output, a few thousand weights;</li>
 *   <li>{@link #medianSmooth} for frame smoothing, and the metrics {@link #f1} and
 *       {@link #auc}.</li>
 * </ul>
 * Paper: <a href="https://doi.org/10.1109/97.736233">doi:10.1109/97.736233</a>.
 */
public final class VoiceActivity {

    private VoiceActivity() {
    }

    /**
     * Frame log energies (dB).
     *
     * @param wave the samples
     * @param hop samples per frame
     * @return per frame {@code 10 log10(mean square + 1e-10)}
     */
    public static double[] logEnergy(float[] wave, int hop) {
        int frames = wave.length / hop;
        double[] e = new double[frames];
        for (int t = 0; t < frames; t++) {
            double s = 0;
            for (int i = t * hop; i < (t + 1) * hop; i++) {
                s += wave[i] * wave[i];
            }
            e[t] = 10 * Math.log10(s / hop + 1e-10);
        }
        return e;
    }

    /**
     * Zero-crossing rate per frame.
     *
     * @param wave the samples
     * @param hop samples per frame
     * @return the fraction of sign changes
     */
    public static double[] zeroCrossings(float[] wave, int hop) {
        int frames = wave.length / hop;
        double[] z = new double[frames];
        for (int t = 0; t < frames; t++) {
            int c = 0;
            for (int i = t * hop + 1; i < (t + 1) * hop; i++) {
                if ((wave[i] >= 0) != (wave[i - 1] >= 0)) {
                    c++;
                }
            }
            z[t] = c / (double) (hop - 1);
        }
        return z;
    }

    private static double percentile(double[] v, double q) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s[Math.min(s.length - 1, (int) (q * s.length))];
    }

    /**
     * The energy detector's score: frame energy over the adaptive floor (dB), minus a penalty
     * of 10 dB for frames whose zero-crossing rate exceeds {@code 0.5} (noise-like).
     *
     * @param energiesDb frame energies in dB (possibly from another front end)
     * @param zcr zero-crossing rates, or {@code null}
     * @return per-frame scores; speech where the score exceeds a margin such as 6 dB
     */
    public static double[] energy(double[] energiesDb, double[] zcr) {
        double floor = percentile(energiesDb, 0.1);
        double[] s = new double[energiesDb.length];
        for (int t = 0; t < s.length; t++) {
            s[t] = energiesDb[t] - floor - (zcr != null && zcr[t] > 0.5 ? 10 : 0);
        }
        return s;
    }

    /**
     * Sohn's likelihood-ratio scores.
     *
     * @param wave the samples
     * @param hop samples per frame (the FFT has twice as many)
     * @return per frame the mean log likelihood ratio
     */
    public static double[] sohn(float[] wave, int hop) {
        int frames = wave.length / hop;
        int bins = hop + 1;
        float[] p = Signal.power(Signal.stft(Var.of(Tensor.of(Shape.of(wave.length), wave)), 2 * hop, hop)).value()
                .toArray();
        double[][] power = new double[frames][bins];
        double[] total = new double[frames];
        for (int t = 0; t < frames; t++) {
            for (int k = 0; k < bins; k++) {
                power[t][k] = p[t * bins + k] + 1e-10;
                total[t] += power[t][k];
            }
        }
        double cut = percentile(total, 0.1);
        double[] noise = new double[bins];
        int quiet = 0;
        for (int t = 0; t < frames; t++) {
            if (total[t] <= cut) {
                quiet++;
                for (int k = 0; k < bins; k++) {
                    noise[k] += power[t][k];
                }
            }
        }
        for (int k = 0; k < bins; k++) {
            noise[k] = noise[k] / Math.max(1, quiet) + 1e-10;
        }
        double alpha = 0.98;
        double[] prevClean = new double[bins];
        double[] out = new double[frames];
        for (int t = 0; t < frames; t++) {
            double sum = 0;
            double[] clean = new double[bins];
            for (int k = 0; k < bins; k++) {
                double gamma = Math.min(power[t][k] / noise[k], 1e4);
                double xi = alpha * prevClean[k] / noise[k] + (1 - alpha) * Math.max(gamma - 1, 0);
                xi = Math.max(xi, 1e-3);
                sum += gamma * xi / (1 + xi) - Math.log(1 + xi);
                double gain = xi / (1 + xi);
                clean[k] = gain * gain * power[t][k];
            }
            prevClean = clean;
            out[t] = sum / bins;
            if (out[t] < 0.15) {
                for (int k = 0; k < bins; k++) {
                    noise[k] = 0.95 * noise[k] + 0.05 * power[t][k];
                }
            }
        }
        return out;
    }

    /**
     * HMM hangover: fits a two-state Gaussian-emission HMM to the score sequence (states
     * started at its 20th and 90th percentiles, transitions staying with probability
     * {@code stay}) by Baum-Welch and returns the posterior of the higher-mean state.
     *
     * @param scores per-frame scores
     * @param stay the initial self-transition probability (such as 0.95)
     * @return per-frame speech probabilities
     */
    public static double[] hmmHangover(double[] scores, double stay) {
        double lo = percentile(scores, 0.2);
        double hi = percentile(scores, 0.9);
        double var = 0;
        double mean = Arrays.stream(scores).average().orElse(0);
        for (double s : scores) {
            var += (s - mean) * (s - mean) / scores.length;
        }
        var = Math.max(var / 4, 1e-4);
        HiddenMarkovModel hmm = HiddenMarkovModel.gaussian(new double[] {0.5, 0.5},
                new double[][] {{stay, 1 - stay}, {1 - stay, stay}}, new double[] {lo, hi}, new double[] {var, var});
        hmm = hmm.baumWelch(List.of(scores), 8);
        double[][] post = hmm.posteriors(scores);
        int speech = hmm.means()[1] >= hmm.means()[0] ? 1 : 0;
        double[] out = new double[scores.length];
        for (int t = 0; t < out.length; t++) {
            out[t] = post[t][speech];
        }
        return out;
    }

    /**
     * Median smoothing of scores.
     *
     * @param scores per-frame scores
     * @param radius the window radius
     * @return smoothed scores
     */
    public static double[] medianSmooth(double[] scores, int radius) {
        double[] out = new double[scores.length];
        double[] w = new double[2 * radius + 1];
        for (int t = 0; t < scores.length; t++) {
            for (int j = -radius; j <= radius; j++) {
                w[j + radius] = scores[Math.min(scores.length - 1, Math.max(0, t + j))];
            }
            double[] s = w.clone();
            Arrays.sort(s);
            out[t] = s[radius];
        }
        return out;
    }

    /**
     * Thresholds scores.
     *
     * @param scores per-frame scores
     * @param threshold the threshold
     * @return the decisions
     */
    public static boolean[] decide(double[] scores, double threshold) {
        boolean[] d = new boolean[scores.length];
        for (int t = 0; t < d.length; t++) {
            d[t] = scores[t] > threshold;
        }
        return d;
    }

    /**
     * Frame F1 of speech decisions.
     *
     * @param predicted the decisions
     * @param truth the labels (the shorter length is used)
     * @return the F1 score
     */
    public static double f1(boolean[] predicted, boolean[] truth) {
        int n = Math.min(predicted.length, truth.length);
        int tp = 0;
        int fp = 0;
        int fn = 0;
        for (int t = 0; t < n; t++) {
            if (predicted[t] && truth[t]) {
                tp++;
            } else if (predicted[t]) {
                fp++;
            } else if (truth[t]) {
                fn++;
            }
        }
        return 2.0 * tp / Math.max(1, 2 * tp + fp + fn);
    }

    /**
     * Area under the ROC curve (the probability that a random speech frame outscores a random
     * non-speech frame, ties counting half).
     *
     * @param scores per-frame scores
     * @param truth the labels
     * @return the AUC
     */
    public static double auc(double[] scores, boolean[] truth) {
        int n = Math.min(scores.length, truth.length);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        double rankSum = 0;
        int pos = 0;
        int i = 0;
        while (i < n) {
            int j = i;
            while (j + 1 < n && scores[order[j + 1]] == scores[order[i]]) {
                j++;
            }
            double rank = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++) {
                if (truth[order[k]]) {
                    rankSum += rank;
                    pos++;
                }
            }
            i = j + 1;
        }
        int neg = n - pos;
        if (pos == 0 || neg == 0) {
            return 0.5;
        }
        return (rankSum - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }

    /**
     * A small convolutional-recurrent detector: {@code [batch, frames, bands]} log-mel frames to
     * per-frame speech logits {@code [batch, frames]}.
     */
    public static final class Crnn implements Layer {
        private final Var conv;
        private final Var convBias;
        private final Layer gru;
        private final Var out;
        private final Var outBias;
        private final int channels;

        /**
         * Creates the detector.
         *
         * @param bands input bands
         * @param channels convolution channels and GRU units
         * @param random the source of initial weights
         */
        public Crnn(int bands, int channels, SplittableRandom random) {
            this.channels = channels;
            conv = Var.param(Tensor.normal(Shape.of(channels, bands, 3), 0, (float) Math.sqrt(2.0 / (3 * bands)), random));
            convBias = Var.param(Tensor.zeros(Shape.of(channels)));
            gru = Recurrent.gru(channels).returnSequences(true).create(Shape.of(8, channels), random);
            out = Var.param(Tensor.normal(Shape.of(channels, 1), 0, (float) Math.sqrt(1.0 / channels), random));
            outBias = Var.param(Tensor.zeros(Shape.of(1)));
        }

        @Override
        public Var forward(Var input) {
            int b = input.shape().dim(0);
            int t = input.shape().dim(1);
            Var h = Functional.conv(input.permute(0, 2, 1), conv, convBias, new int[] {1}, new int[] {1},
                    new int[] {1}, new int[] {1}, 1).relu().permute(0, 2, 1);
            Var seq = gru.forward(h).add(h);
            return seq.matmul(out).add(outBias).reshape(b, t);
        }

        @Override
        public Shape outputShape(Shape input) {
            return Shape.of(input.dim(0));
        }

        /**
         * The logistic loss against frame labels.
         *
         * @param features {@code [b, frames, bands]}
         * @param labels {@code [b, frames]} in {0, 1}
         * @return the mean loss
         */
        public Var loss(Tensor features, Tensor labels) {
            Var z = forward(Var.of(features));
            return z.softplus().sub(z.mul(labels)).mean();
        }

        /**
         * Speech probabilities for one utterance.
         *
         * @param features {@code [frames][bands]}
         * @return per frame the probability
         */
        public double[] probabilities(float[][] features) {
            float[] p = Var.noGrad(() -> forward(Var.of(SpeechFeatures.batch(List.<float[][]>of(features)))).sigmoid()).value()
                    .toArray();
            return SpeechFeatures.toDouble(p);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(conv, convBias));
            p.addAll(gru.parameters());
            p.add(out);
            p.add(outBias);
            return p;
        }

        /** @return the weight size in kB at 32 bits per weight */
        public double sizeKb() {
            long n = 0;
            for (Var v : parameters()) {
                n += v.shape().size();
            }
            return n * 4 / 1024.0;
        }

        @Override
        public String toString() {
            return "VAD-CRNN(" + channels + ")";
        }
    }
}
