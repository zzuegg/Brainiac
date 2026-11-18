package io.github.zzuegg.brainiac.experiments.context;

import io.github.zzuegg.brainiac.experimental.llm.PositionBiasCalibration;
import io.github.zzuegg.brainiac.experimental.llm.RopeScaling;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Lost in the middle and its calibration (8.4.11, {@link PositionBiasCalibration}), on
 * multi-document key-value QA read by a bank of RoPE retrieval heads. Ten documents of twelve
 * tokens precede a question; the answer document's keys carry the question's content
 * direction, the others random content. As in trained LLMs, keys and queries also share a
 * common offset per head; under RoPE that offset gives every position a content-independent
 * attention bias that depends on the distance to the question. The model answers with the
 * document that receives the most attention (summed over its tokens and averaged over four
 * heads). Rows, with the answer document at each of the ten positions:
 * <ul>
 * <li>the uncalibrated heads (RoPE base 500);</li>
 * <li>found-in-the-middle: the bias is measured with irrelevant dummy documents at every
 *     position and subtracted from each document's attention;</li>
 * <li>Ms-PoE: per-head position ratios 1.2 to 1.8;</li>
 * <li>YaRN ({@link RopeScaling}, factor 2 over a 64-token original context) at the same
 *     length.</li>
 * </ul>
 * Reported: accuracy by answer position and the middle-minus-ends gap (positions 3 to 6 against
 * 0, 1, 8 and 9).
 */
public final class PositionBiasSweep {

    private static final int DOCS = 10;
    private static final int LEN = 12;
    private static final int DIM = 32;
    private static final int HEADS = 4;
    private static final double THETA = 500;

    private PositionBiasSweep() {
    }

    /**
     * One method.
     *
     * @param accuracy accuracy per answer position
     * @param gap mean middle accuracy minus mean end accuracy
     */
    public record Curve(double[] accuracy, double gap) {
    }

    private static final class Heads {
        final float[][] keyOffset = new float[HEADS][DIM];
        final float[][] queryOffset = new float[HEADS][DIM];

        Heads(SplittableRandom r) {
            for (int h = 0; h < HEADS; h++) {
                keyOffset[h] = gaussian(r, 1.0);
                queryOffset[h] = gaussian(r, 1.0);
            }
        }
    }

    private static float[] gaussian(SplittableRandom r, double std) {
        float[] v = new float[DIM];
        for (int c = 0; c < DIM; c++) {
            v[c] = (float) (r.nextGaussian() * std / Math.sqrt(DIM));
        }
        return v;
    }

    /**
     * The attention mass every document receives, averaged over heads.
     *
     * @param answerAt the answer document's position, or -1 for none (dummy trials)
     */
    private static double[] masses(Heads heads, int answerAt, double[] freq, PositionBiasCalibration msPoe,
            SplittableRandom r) {
        float[] content = gaussian(r, 1.0);
        int n = DOCS * LEN;
        float[][] keyContent = new float[n][];
        for (int t = 0; t < n; t++) {
            keyContent[t] = gaussian(r, 1.0);
            if (t / LEN == answerAt) {
                for (int c = 0; c < DIM; c++) {
                    keyContent[t][c] += 0.5f * content[c];
                }
            }
        }
        double[] mass = new double[DOCS];
        for (int h = 0; h < HEADS; h++) {
            float[] q = new float[DIM];
            for (int c = 0; c < DIM; c++) {
                q[c] = 6 * (content[c] + heads.queryOffset[h][c]);
            }
            double qPos = msPoe == null ? n : msPoe.position(n, h);
            float[] qr = PositionBiasCalibration.rotate(q, qPos, freq);
            double[] s = new double[n];
            double max = Double.NEGATIVE_INFINITY;
            for (int t = 0; t < n; t++) {
                float[] k = new float[DIM];
                for (int c = 0; c < DIM; c++) {
                    k[c] = 6 * (keyContent[t][c] + heads.keyOffset[h][c]);
                }
                double kPos = msPoe == null ? t : msPoe.position(t, h);
                float[] kr = PositionBiasCalibration.rotate(k, kPos, freq);
                double d = 0;
                for (int c = 0; c < DIM; c++) {
                    d += qr[c] * kr[c];
                }
                s[t] = d / Math.sqrt(DIM);
                max = Math.max(max, s[t]);
            }
            double z = 0;
            for (int t = 0; t < n; t++) {
                s[t] = Math.exp(s[t] - max);
                z += s[t];
            }
            for (int t = 0; t < n; t++) {
                mass[t / LEN] += s[t] / z / HEADS;
            }
        }
        return mass;
    }

    /**
     * Runs the sweep.
     *
     * @param trials questions per answer position and seed
     * @param seeds the seeds (each draws its own heads)
     * @return method to curve
     */
    public static Map<String, Curve> run(int trials, long[] seeds) {
        double[] base = new io.github.zzuegg.brainiac.llm.Rope(new io.github.zzuegg.brainiac.llm.Rope.Settings(DIM,
                THETA, "default", 1, 1, 4, 2048, null)).frequencies();
        double[] yarn = RopeScaling.of(DIM, THETA, RopeScaling.Method.YARN, 2, 64).frequencies();
        PositionBiasCalibration msPoe = PositionBiasCalibration.multiScalePositions(HEADS, 1.2, 1.8, null);
        String[] names = {"uncalibrated", "found-in-the-middle", "Ms-PoE", "YaRN (factor 2)"};
        Map<String, double[]> acc = new LinkedHashMap<>();
        for (String name : names) {
            acc.put(name, new double[DOCS]);
        }
        for (long seed : seeds) {
            SplittableRandom r = new SplittableRandom(seed);
            Heads heads = new Heads(r);
            double[][] dummy = new double[200][];
            for (int i = 0; i < dummy.length; i++) {
                dummy[i] = masses(heads, -1, base, null, r);
            }
            double[] bias = PositionBiasCalibration.foundInTheMiddle(dummy);
            for (int pos = 0; pos < DOCS; pos++) {
                for (int t = 0; t < trials; t++) {
                    long s = r.nextLong();
                    double[] m = masses(heads, pos, base, null, new SplittableRandom(s));
                    acc.get("uncalibrated")[pos] += argmax(m) == pos ? 1 : 0;
                    acc.get("found-in-the-middle")[pos] += argmax(PositionBiasCalibration.calibrate(m, bias)) == pos
                            ? 1 : 0;
                    acc.get("Ms-PoE")[pos] += argmax(masses(heads, pos, base, msPoe, new SplittableRandom(s))) == pos
                            ? 1 : 0;
                    acc.get("YaRN (factor 2)")[pos] += argmax(masses(heads, pos, yarn, null, new SplittableRandom(s)))
                            == pos ? 1 : 0;
                }
            }
        }
        Map<String, Curve> out = new LinkedHashMap<>();
        for (String name : names) {
            double[] a = acc.get(name);
            for (int p = 0; p < DOCS; p++) {
                a[p] /= (double) trials * seeds.length;
            }
            double middle = (a[3] + a[4] + a[5] + a[6]) / 4;
            double ends = (a[0] + a[1] + a[8] + a[9]) / 4;
            out.put(name, new Curve(a, middle - ends));
        }
        return out;
    }

    private static int argmax(double[] x) {
        int b = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[i] > x[b]) {
                b = i;
            }
        }
        return b;
    }
}
