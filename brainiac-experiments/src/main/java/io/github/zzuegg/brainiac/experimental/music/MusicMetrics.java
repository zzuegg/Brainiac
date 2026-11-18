package io.github.zzuegg.brainiac.experimental.music;

import java.util.ArrayList;
import java.util.List;

/**
 * Evaluation measures for music information retrieval, following {@code mir_eval}'s
 * definitions where one exists: melody (raw pitch and raw chroma accuracy within 50 cents,
 * voicing F1, octave-error rate), onset and beat F1 with a tolerance window, the beat
 * continuity measure CMLt, weighted chord symbol recall and directional-Hamming segmentation
 * quality, and event-based and segment-based sound event detection scores.
 */
public final class MusicMetrics {

    private MusicMetrics() {
    }

    /**
     * Melody scores.
     *
     * @param rawPitch share of voiced reference frames whose estimate is within 50 cents
     * @param rawChroma the same ignoring octave errors
     * @param voicingF1 F1 of the voiced/unvoiced decision (voiced is positive)
     * @param octaveErrors share of voiced reference frames off by an octave (within 50 cents
     *     of 2x or 0.5x) among those that miss raw pitch
     */
    public record Pitch(double rawPitch, double rawChroma, double voicingF1, double octaveErrors) {
    }

    /**
     * @param a a frequency
     * @param b another
     * @return their distance in cents
     */
    public static double cents(double a, double b) {
        return 1200 * Math.log(a / b) / Math.log(2);
    }

    /**
     * Scores a pitch track. Raw pitch counts the estimate of every voiced reference frame
     * even if the estimator marked it unvoiced (as {@code mir_eval} does with the frequency
     * it reports), so {@code estF0} must carry a candidate on every frame.
     *
     * @param refF0 reference F0 per frame, 0 for unvoiced
     * @param estF0 estimated F0 per frame (a candidate even where unvoiced; 0 if none)
     * @param estVoiced estimated voicing per frame
     * @return the scores
     */
    public static Pitch pitch(float[] refF0, float[] estF0, boolean[] estVoiced) {
        int n = Math.min(refF0.length, estF0.length);
        int voiced = 0;
        int rp = 0;
        int rc = 0;
        int octave = 0;
        int tp = 0;
        int fp = 0;
        int fn = 0;
        for (int i = 0; i < n; i++) {
            boolean rv = refF0[i] > 0;
            boolean ev = estVoiced[i];
            tp += rv && ev ? 1 : 0;
            fp += !rv && ev ? 1 : 0;
            fn += rv && !ev ? 1 : 0;
            if (!rv) {
                continue;
            }
            voiced++;
            if (estF0[i] <= 0) {
                continue;
            }
            double c = cents(estF0[i], refF0[i]);
            if (Math.abs(c) <= 50) {
                rp++;
                rc++;
            } else {
                double folded = Math.abs(c - 1200 * Math.round(c / 1200));
                if (folded <= 50) {
                    rc++;
                    octave++;
                }
            }
        }
        double f1 = 2.0 * tp / Math.max(1, 2 * tp + fp + fn);
        return new Pitch(rp / (double) Math.max(1, voiced), rc / (double) Math.max(1, voiced), f1,
                octave / (double) Math.max(1, voiced - rp));
    }

    /**
     * F1 of event times matched one-to-one within a tolerance (greedy in time order, as
     * {@code mir_eval.onset}).
     *
     * @param reference reference times in seconds, sorted
     * @param estimate estimated times, sorted
     * @param tolerance the window half-width in seconds
     * @return {@code {precision, recall, f1}}
     */
    public static double[] matchF1(double[] reference, double[] estimate, double tolerance) {
        boolean[] used = new boolean[estimate.length];
        int hit = 0;
        for (double r : reference) {
            int best = -1;
            double bestD = tolerance;
            for (int j = 0; j < estimate.length; j++) {
                double d = Math.abs(estimate[j] - r);
                if (!used[j] && d <= bestD) {
                    best = j;
                    bestD = d;
                }
            }
            if (best >= 0) {
                used[best] = true;
                hit++;
            }
        }
        double p = estimate.length == 0 ? 0 : hit / (double) estimate.length;
        double rc = reference.length == 0 ? 0 : hit / (double) reference.length;
        return new double[] {p, rc, p + rc == 0 ? 0 : 2 * p * rc / (p + rc)};
    }

    /**
     * CMLt: the share of reference beats tracked continuously at the correct metrical level
     * (a beat counts when it and the previous estimate are within 17.5% of the inter-beat
     * interval of consecutive reference beats, and the estimated interval matches the
     * reference interval within 17.5%); the total over all continuous segments, as in
     * Davies et al.'s evaluation toolbox.
     *
     * @param reference reference beats, sorted
     * @param estimate estimated beats, sorted
     * @return CMLt in {@code [0, 1]}
     */
    public static double cmlt(double[] reference, double[] estimate) {
        if (reference.length < 2 || estimate.length < 2) {
            return 0;
        }
        int correct = 0;
        for (int i = 1; i < reference.length; i++) {
            double ibi = reference[i] - reference[i - 1];
            int j = nearest(estimate, reference[i]);
            int jp = nearest(estimate, reference[i - 1]);
            boolean phase = Math.abs(estimate[j] - reference[i]) <= 0.175 * ibi
                    && Math.abs(estimate[jp] - reference[i - 1]) <= 0.175 * ibi;
            boolean period = j == jp + 1 && Math.abs((estimate[j] - estimate[jp]) - ibi) <= 0.175 * ibi;
            correct += phase && period ? 1 : 0;
        }
        return correct / (double) reference.length;
    }

    private static int nearest(double[] t, double x) {
        int best = 0;
        for (int i = 1; i < t.length; i++) {
            if (Math.abs(t[i] - x) < Math.abs(t[best] - x)) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Weighted chord symbol recall: the share of frames (so of time) whose chord label is
     * right, over frames whose reference has a chord or is "no chord".
     *
     * @param reference per-frame labels
     * @param estimate per-frame labels
     * @return the recall
     */
    public static double chordRecall(int[] reference, int[] estimate) {
        int n = Math.min(reference.length, estimate.length);
        int hit = 0;
        for (int i = 0; i < n; i++) {
            hit += reference[i] == estimate[i] ? 1 : 0;
        }
        return hit / (double) Math.max(1, n);
    }

    private static List<int[]> segments(int[] labels) {
        List<int[]> s = new ArrayList<>();
        int start = 0;
        for (int i = 1; i <= labels.length; i++) {
            if (i == labels.length || labels[i] != labels[start]) {
                s.add(new int[] {start, i});
                start = i;
            }
        }
        return s;
    }

    private static double directionalHamming(List<int[]> a, List<int[]> b, int n) {
        double missed = 0;
        for (int[] sa : a) {
            int best = 0;
            for (int[] sb : b) {
                best = Math.max(best, Math.max(0, Math.min(sa[1], sb[1]) - Math.max(sa[0], sb[0])));
            }
            missed += (sa[1] - sa[0]) - best;
        }
        return missed / n;
    }

    /**
     * Segmentation quality {@code 1 - max(under, over)} from the directional Hamming
     * distances between the reference and estimated segmentations (Harte's measure).
     *
     * @param reference per-frame labels
     * @param estimate per-frame labels
     * @return the quality in {@code [0, 1]}
     */
    public static double segmentation(int[] reference, int[] estimate) {
        int n = Math.min(reference.length, estimate.length);
        List<int[]> r = segments(java.util.Arrays.copyOf(reference, n));
        List<int[]> e = segments(java.util.Arrays.copyOf(estimate, n));
        return 1 - Math.max(directionalHamming(r, e, n), directionalHamming(e, r, n));
    }

    /**
     * The events of one class as {@code {onsetFrame, offsetFrame}} runs.
     *
     * @param active per-frame activity
     * @return the runs
     */
    public static List<int[]> runs(boolean[] active) {
        List<int[]> out = new ArrayList<>();
        int start = -1;
        for (int i = 0; i <= active.length; i++) {
            boolean on = i < active.length && active[i];
            if (on && start < 0) {
                start = i;
            } else if (!on && start >= 0) {
                out.add(new int[] {start, i});
                start = -1;
            }
        }
        return out;
    }

    /**
     * Event-based F1 over all classes: an estimated event matches a reference event of the
     * same class whose onset is within the collar (onset-only, as DCASE's event-based
     * metric with a 200 ms collar and no offset condition).
     *
     * @param reference per frame and class the reference activity
     * @param estimate per frame and class the estimated activity
     * @param collarFrames the onset collar in frames
     * @return the micro-averaged F1
     */
    public static double eventF1(boolean[][] reference, boolean[][] estimate, int collarFrames) {
        int classes = reference[0].length;
        int tp = 0;
        int nr = 0;
        int ne = 0;
        for (int c = 0; c < classes; c++) {
            List<int[]> r = runs(column(reference, c));
            List<int[]> e = runs(column(estimate, c));
            nr += r.size();
            ne += e.size();
            boolean[] used = new boolean[e.size()];
            for (int[] rr : r) {
                for (int j = 0; j < e.size(); j++) {
                    if (!used[j] && Math.abs(e.get(j)[0] - rr[0]) <= collarFrames) {
                        used[j] = true;
                        tp++;
                        break;
                    }
                }
            }
        }
        return 2.0 * tp / Math.max(1, nr + ne);
    }

    /**
     * Segment-based error rate {@code (S + D + I) / N} over fixed segments (DCASE): per
     * segment, a class is active if it is active in any frame of it; substitutions pair
     * misses with false alarms in the same segment.
     *
     * @param reference per frame and class the reference activity
     * @param estimate per frame and class the estimated activity
     * @param segmentFrames frames per segment (100 for 1 s at 10 ms frames)
     * @return the error rate
     */
    public static double segmentErrorRate(boolean[][] reference, boolean[][] estimate, int segmentFrames) {
        int classes = reference[0].length;
        double s = 0;
        double d = 0;
        double ins = 0;
        double n = 0;
        int frames = Math.min(reference.length, estimate.length);
        for (int start = 0; start < frames; start += segmentFrames) {
            int fn = 0;
            int fp = 0;
            for (int c = 0; c < classes; c++) {
                boolean r = false;
                boolean e = false;
                for (int f = start; f < Math.min(frames, start + segmentFrames); f++) {
                    r |= reference[f][c];
                    e |= estimate[f][c];
                }
                n += r ? 1 : 0;
                fn += r && !e ? 1 : 0;
                fp += !r && e ? 1 : 0;
            }
            int sub = Math.min(fn, fp);
            s += sub;
            d += fn - sub;
            ins += fp - sub;
        }
        return (s + d + ins) / Math.max(1, n);
    }

    private static boolean[] column(boolean[][] m, int c) {
        boolean[] out = new boolean[m.length];
        for (int i = 0; i < m.length; i++) {
            out[i] = m[i][c];
        }
        return out;
    }

    /**
     * Picks peaks of a detection function: local maxima within {@code ±window} frames that
     * exceed the moving mean over {@code ±avgWindow} frames plus {@code delta}, at least
     * {@code minGap} frames apart.
     *
     * @param f the detection function
     * @param window the local-maximum half-width
     * @param avgWindow the moving-average half-width
     * @param delta the threshold offset
     * @param minGap the smallest distance between peaks
     * @return the peak frames
     */
    public static int[] pickPeaks(float[] f, int window, int avgWindow, double delta, int minGap) {
        List<Integer> peaks = new ArrayList<>();
        for (int t = 0; t < f.length; t++) {
            boolean max = true;
            for (int j = Math.max(0, t - window); j <= Math.min(f.length - 1, t + window); j++) {
                if (f[j] > f[t]) {
                    max = false;
                    break;
                }
            }
            if (!max) {
                continue;
            }
            double mean = 0;
            int cnt = 0;
            for (int j = Math.max(0, t - avgWindow); j <= Math.min(f.length - 1, t + avgWindow); j++) {
                mean += f[j];
                cnt++;
            }
            if (f[t] >= mean / cnt + delta && (peaks.isEmpty() || t - peaks.get(peaks.size() - 1) >= minGap)) {
                peaks.add(t);
            }
        }
        return peaks.stream().mapToInt(Integer::intValue).toArray();
    }
}
