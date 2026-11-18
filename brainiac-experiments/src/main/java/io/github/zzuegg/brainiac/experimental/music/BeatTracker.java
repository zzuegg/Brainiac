package io.github.zzuegg.brainiac.experimental.music;

/**
 * Ellis's dynamic-programming beat tracker. A global tempo is estimated from the
 * autocorrelation of the onset envelope, weighted by a log-Gaussian tempo prior around
 * 120 BPM (one octave wide); beats are then the frames maximising
 *
 * <pre>{@code C(t) = O(t) + max_{t - 2P <= s <= t - P/2} ( C(s) - alpha (log((t - s) / P))^2 )}</pre>
 *
 * which trades the onset strength {@code O} against deviation of every inter-beat interval
 * from the period {@code P}, found by backtracking from the best-scoring final beat. The
 * fixed period is its strength on steady tempi and its weakness when the tempo drifts.
 *
 * <p>Paper: Ellis, "Beat Tracking by Dynamic Programming", Journal of New Music Research
 * 36(1), 2007, <a href="https://doi.org/10.1080/09298210701653344">doi:10.1080/09298210701653344</a>.
 */
public final class BeatTracker {

    private final double tightness;
    private final double priorBpm;

    /**
     * Creates a tracker.
     *
     * @param tightness the weight {@code alpha} of tempo deviations (Ellis uses 100)
     * @param priorBpm the centre of the tempo prior
     */
    public BeatTracker(double tightness, double priorBpm) {
        this.tightness = tightness;
        this.priorBpm = priorBpm;
    }

    /**
     * The tempo period of an onset envelope.
     *
     * @param onset the onset envelope
     * @param fps frames per second
     * @return the period in frames
     */
    public double period(float[] onset, double fps) {
        int minLag = (int) Math.floor(fps * 60 / 240);
        int maxLag = (int) Math.ceil(fps * 60 / 40);
        double best = Double.NEGATIVE_INFINITY;
        int arg = minLag;
        double mean = 0;
        for (float v : onset) {
            mean += v;
        }
        mean /= onset.length;
        for (int lag = minLag; lag <= Math.min(maxLag, onset.length - 1); lag++) {
            double s = 0;
            for (int t = lag; t < onset.length; t++) {
                s += (onset[t] - mean) * (onset[t - lag] - mean);
            }
            s /= onset.length - lag;
            double bpm = 60 * fps / lag;
            double prior = Math.exp(-0.5 * Math.pow(Math.log(bpm / priorBpm) / Math.log(2), 2));
            if (s * prior > best) {
                best = s * prior;
                arg = lag;
            }
        }
        return arg;
    }

    /**
     * Tracks beats.
     *
     * @param onset the onset envelope (any scale; standardised inside)
     * @param fps frames per second
     * @return beat times in seconds
     */
    public double[] beats(float[] onset, double fps) {
        float[] o = OnsetDetection.standardise(onset);
        double p = period(o, fps);
        int n = o.length;
        double[] score = new double[n];
        int[] back = new int[n];
        for (int t = 0; t < n; t++) {
            double best = 0;
            int arg = -1;
            int lo = (int) Math.max(0, Math.round(t - 2 * p));
            int hi = (int) Math.round(t - p / 2);
            for (int s = lo; s <= hi; s++) {
                double v = score[s] - tightness * Math.pow(Math.log((t - s) / p), 2);
                if (arg < 0 || v > best) {
                    best = v;
                    arg = s;
                }
            }
            score[t] = o[t] + (arg >= 0 ? best : 0);
            back[t] = arg;
        }
        int last = n - 1;
        for (int t = Math.max(0, n - (int) Math.ceil(p)); t < n; t++) {
            if (score[t] > score[last]) {
                last = t;
            }
        }
        java.util.List<Double> beats = new java.util.ArrayList<>();
        for (int t = last; t >= 0; t = back[t]) {
            beats.add(0, t / fps);
            if (back[t] < 0) {
                break;
            }
        }
        return beats.stream().mapToDouble(Double::doubleValue).toArray();
    }
}
