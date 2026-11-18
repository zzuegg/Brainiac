package io.github.zzuegg.brainiac.experimental.music;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Shazam-style landmark audio fingerprinting. Peaks of the log-magnitude spectrogram (local
 * maxima of a {@code ±7} bin by {@code ±3} frame neighbourhood that stand above the frame's
 * mean, at most {@link #PEAKS_PER_FRAME} per frame) are paired: each anchor with up to
 * {@link #FAN_OUT} peaks in a target zone 1-32 frames later and within {@code ±48} bins. A
 * pair's hash packs {@code (f1, f2, dt)}; the index maps every hash to the (track, anchor
 * time) occurrences. A query votes for (track, time offset) pairs over all its hashes, and
 * the track with the largest single offset bin wins, because a true match lines its
 * landmarks up at one consistent offset.
 *
 * <p>Exact time differences and exact bins make the hashes robust to noise and filtering but
 * fragile under speed changes, which shift every frequency and stretch every {@code dt}.
 *
 * <p>Paper: Wang, "An Industrial-Strength Audio Search Algorithm", ISMIR 2003.
 */
public final class LandmarkFingerprint {

    /** The STFT size. */
    public static final int N_FFT = 512;
    /** The hop. */
    public static final int HOP = 128;
    /** The most peaks kept per frame. */
    public static final int PEAKS_PER_FRAME = 5;
    /** Partners per anchor. */
    public static final int FAN_OUT = 5;

    private final Map<Integer, List<Long>> index = new HashMap<>();
    private int tracks;

    /** Creates an empty index. */
    public LandmarkFingerprint() {
    }

    /**
     * The spectral peaks of a signal.
     *
     * @param x the signal
     * @return {@code {frame, bin}} pairs in time order
     */
    public static List<int[]> peaks(float[] x) {
        float[][] m = MusicFeatures.magnitude(x, N_FFT, HOP);
        int frames = m.length;
        int bins = m[0].length;
        float[][] l = new float[frames][bins];
        for (int t = 0; t < frames; t++) {
            for (int k = 0; k < bins; k++) {
                l[t][k] = (float) Math.log(1e-6 + m[t][k]);
            }
        }
        List<int[]> out = new ArrayList<>();
        for (int t = 0; t < frames; t++) {
            double mean = 0;
            for (int k = 0; k < bins; k++) {
                mean += l[t][k];
            }
            mean /= bins;
            List<int[]> cand = new ArrayList<>();
            for (int k = 2; k < bins - 1; k++) {
                float v = l[t][k];
                if (v < mean + 2) {
                    continue;
                }
                boolean max = true;
                for (int dt = -3; dt <= 3 && max; dt++) {
                    int tt = t + dt;
                    if (tt < 0 || tt >= frames) {
                        continue;
                    }
                    for (int dk = -7; dk <= 7; dk++) {
                        int kk = k + dk;
                        if (kk >= 0 && kk < bins && (dt != 0 || dk != 0) && l[tt][kk] > v) {
                            max = false;
                            break;
                        }
                    }
                }
                if (max) {
                    cand.add(new int[] {t, k, Float.floatToIntBits(v)});
                }
            }
            cand.sort((a, b) -> Float.compare(Float.intBitsToFloat(b[2]), Float.intBitsToFloat(a[2])));
            for (int i = 0; i < Math.min(PEAKS_PER_FRAME, cand.size()); i++) {
                out.add(new int[] {cand.get(i)[0], cand.get(i)[1]});
            }
        }
        return out;
    }

    /**
     * The landmark hashes of a signal.
     *
     * @param x the signal
     * @return {@code {hash, anchorFrame}} pairs
     */
    public static List<int[]> landmarks(float[] x) {
        List<int[]> p = peaks(x);
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < p.size(); i++) {
            int[] a = p.get(i);
            int paired = 0;
            for (int j = i + 1; j < p.size() && paired < FAN_OUT; j++) {
                int[] b = p.get(j);
                int dt = b[0] - a[0];
                if (dt < 1) {
                    continue;
                }
                if (dt > 32) {
                    break;
                }
                if (Math.abs(b[1] - a[1]) > 48) {
                    continue;
                }
                out.add(new int[] {(a[1] << 15) | (b[1] << 6) | dt, a[0]});
                paired++;
            }
        }
        return out;
    }

    /**
     * Adds a reference recording.
     *
     * @param x its signal
     * @return its track id
     */
    public int add(float[] x) {
        int id = tracks++;
        for (int[] h : landmarks(x)) {
            index.computeIfAbsent(h[0], k -> new ArrayList<>()).add(((long) id << 32) | h[1]);
        }
        return id;
    }

    /**
     * Identifies a query.
     *
     * @param query its signal
     * @return {@code {track, votes}} of the best track, {@code {-1, 0}} without a match
     */
    public int[] identify(float[] query) {
        Map<Long, Integer> votes = new HashMap<>();
        for (int[] h : landmarks(query)) {
            List<Long> hits = index.get(h[0]);
            if (hits == null) {
                continue;
            }
            for (long e : hits) {
                long track = e >>> 32;
                long offset = (int) e - h[1] + 100_000;
                votes.merge((track << 32) | offset, 1, Integer::sum);
            }
        }
        int best = -1;
        int count = 0;
        for (Map.Entry<Long, Integer> e : votes.entrySet()) {
            if (e.getValue() > count) {
                count = e.getValue();
                best = (int) (e.getKey() >>> 32);
            }
        }
        return new int[] {best, count};
    }

    /** @return the number of stored hashes */
    public int size() {
        int n = 0;
        for (List<Long> l : index.values()) {
            n += l.size();
        }
        return n;
    }
}
