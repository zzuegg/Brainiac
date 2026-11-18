package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;

/**
 * Prequential (interleaved test-then-train) evaluation: every example is first predicted, then
 * learned from, so every example is a test example before it is a training one and the whole
 * stream is used for both (Dawid, 1984; Gama et al., 2013). Reports the overall accuracy and log
 * loss, the accuracy over consecutive windows, and the recovery time after each known drift: the
 * samples until the windowed accuracy (over the last {@code window / 2} samples) gets back to
 * within 0.05 of its level just before the drift.
 *
 * <p>Paper: Gama, Sebastião and Rodrigues, "On evaluating stream learning algorithms",
 * Machine Learning 90 (2013).
 */
public final class Prequential {

    /**
     * The evaluation of one learner on one stream.
     *
     * @param accuracy overall prequential accuracy
     * @param logLoss overall prequential log loss
     * @param windows accuracy over consecutive windows
     * @param window the window length
     * @param recovery per drift, samples until accuracy recovered ({@code -1} if never before the
     *     next drift or the end)
     * @param bytes the learner's memory at the end
     * @param seconds wall time
     */
    public record Result(double accuracy, double logLoss, double[] windows, int window, int[] recovery, long bytes,
            double seconds) {

        /** @return the mean recovery time over the drifts that recovered */
        public double meanRecovery() {
            double s = 0;
            int n = 0;
            for (int r : recovery) {
                if (r >= 0) {
                    s += r;
                    n++;
                }
            }
            return n == 0 ? Double.NaN : s / n;
        }
    }

    private Prequential() {
    }

    /**
     * Runs a learner over a stream.
     *
     * @param stream the stream
     * @param learner the learner
     * @param n examples
     * @param window the window for the accuracy curve
     * @return the result
     */
    public static Result evaluate(DriftingStreams.Stream stream, OnlineClassifier learner, int n, int window) {
        return evaluate(DriftingStreams.take(stream, n), stream.driftPoints(), learner, window);
    }

    /**
     * Runs a learner over drawn samples.
     *
     * @param samples the samples, in order
     * @param drifts the drift positions
     * @param learner the learner
     * @param window the window for the accuracy curve
     * @return the result
     */
    public static Result evaluate(DriftingStreams.Sample[] samples, int[] drifts, OnlineClassifier learner,
            int window) {
        int n = samples.length;
        boolean[] correct = new boolean[n];
        double loss = 0;
        int right = 0;
        long start = System.nanoTime();
        for (int i = 0; i < n; i++) {
            DriftingStreams.Sample s = samples[i];
            double[] p = learner.predict(s);
            correct[i] = OnlineClassifier.argmax(p) == s.label();
            right += correct[i] ? 1 : 0;
            loss -= Math.log(Math.max(1e-12, Math.min(1, p[s.label()])));
            learner.learn(s);
        }
        double secs = (System.nanoTime() - start) / 1e9;
        int windows = Math.max(1, n / window);
        double[] curve = new double[windows];
        for (int w = 0; w < windows; w++) {
            int ok = 0;
            for (int i = w * window; i < Math.min(n, (w + 1) * window); i++) {
                ok += correct[i] ? 1 : 0;
            }
            curve[w] = ok / (double) window;
        }
        return new Result(right / (double) n, loss / n, curve, window, recovery(correct, drifts, window / 2),
                learner.bytes(), secs);
    }

    /**
     * Recovery times after each drift.
     *
     * @param correct per-sample correctness
     * @param drifts drift positions
     * @param w the rolling window
     * @return samples to recover per drift, {@code -1} if never
     */
    public static int[] recovery(boolean[] correct, int[] drifts, int w) {
        int n = correct.length;
        int[] prefix = new int[n + 1];
        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + (correct[i] ? 1 : 0);
        }
        int[] out = new int[drifts.length];
        for (int d = 0; d < drifts.length; d++) {
            int p = drifts[d];
            if (p - w < 0 || p >= n) {
                out[d] = -1;
                continue;
            }
            double before = (prefix[p] - prefix[p - w]) / (double) w;
            int end = d + 1 < drifts.length ? Math.min(n, drifts[d + 1]) : n;
            out[d] = -1;
            for (int t = p + w; t <= end; t++) {
                double now = (prefix[t] - prefix[t - w]) / (double) w;
                if (now >= before - 0.05) {
                    out[d] = t - p;
                    break;
                }
            }
        }
        return out;
    }
}
