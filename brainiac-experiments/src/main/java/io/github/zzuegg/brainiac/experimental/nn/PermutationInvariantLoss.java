package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.ArrayList;
import java.util.List;

/**
 * Utterance-level permutation-invariant training (uPIT) with the scale-invariant
 * signal-to-noise ratio. A separator's outputs come in no particular order, so the loss
 * of each utterance is taken under the assignment of outputs to reference sources that
 * scores best (all {@code n!} permutations, fine for the usual two or three sources):
 * {@code -mean_b max_pi mean_i SI-SNR(est_{pi(i)}, ref_i)}. SI-SNR projects the zero-mean
 * estimate onto the zero-mean reference, {@code s_t = <e, r> r / |r|^2}, and returns
 * {@code 10 log10(|s_t|^2 / |e - s_t|^2)}, so gain (and sign) do not matter.
 *
 * <p>Papers: Kolbaek, Yu, Tan, Jensen, "Multi-talker Speech Separation with Utterance-level
 * Permutation Invariant Training" (2017), <a href="https://arxiv.org/abs/1703.06284">arXiv:1703.06284</a>;
 * SI-SNR: Le Roux et al., "SDR - half-baked or well done?" (ICASSP 2019).
 */
public final class PermutationInvariantLoss {

    private static final float LN10 = (float) Math.log(10);

    private PermutationInvariantLoss() {
    }

    /**
     * Differentiable SI-SNR.
     *
     * @param estimate {@code [b, samples]}
     * @param reference {@code [b, samples]}
     * @return {@code [b]} in dB
     */
    public static Var siSnr(Var estimate, Var reference) {
        Var e = estimate.sub(estimate.mean(-1, true));
        Var r = reference.sub(reference.mean(-1, true));
        Var dot = e.mul(r).sum(-1, true);
        Var energy = r.square().sum(-1, true).add(1e-8f);
        Var target = r.mul(dot.div(energy));
        Var noise = e.sub(target);
        Var ratio = target.square().sum(-1, false).add(1e-8f).div(noise.square().sum(-1, false).add(1e-8f));
        return ratio.log().mul(10f / LN10);
    }

    /**
     * SI-SNR of plain arrays.
     *
     * @param estimate samples
     * @param reference samples of the same length
     * @return dB
     */
    public static double siSnr(float[] estimate, float[] reference) {
        int n = reference.length;
        double me = 0;
        double mr = 0;
        for (int i = 0; i < n; i++) {
            me += estimate[i] / (double) n;
            mr += reference[i] / (double) n;
        }
        double dot = 0;
        double energy = 1e-8;
        for (int i = 0; i < n; i++) {
            dot += (estimate[i] - me) * (reference[i] - mr);
            energy += (reference[i] - mr) * (reference[i] - mr);
        }
        double t2 = 1e-8;
        double n2 = 1e-8;
        for (int i = 0; i < n; i++) {
            double t = (reference[i] - mr) * dot / energy;
            double e = estimate[i] - me - t;
            t2 += t * t;
            n2 += e * e;
        }
        return 10 * Math.log10(t2 / n2);
    }

    /**
     * All permutations of {@code 0..n-1}.
     *
     * @param n the size
     * @return the permutations, identity first
     */
    public static List<int[]> permutations(int n) {
        List<int[]> out = new ArrayList<>();
        permute(new int[n], new boolean[n], 0, out);
        return out;
    }

    private static void permute(int[] cur, boolean[] used, int depth, List<int[]> out) {
        if (depth == cur.length) {
            out.add(cur.clone());
            return;
        }
        for (int i = 0; i < cur.length; i++) {
            if (!used[i]) {
                used[i] = true;
                cur[depth] = i;
                permute(cur, used, depth + 1, out);
                used[i] = false;
            }
        }
    }

    /**
     * The uPIT negative SI-SNR loss.
     *
     * @param estimates {@code [b, sources, samples]}
     * @param references {@code [b, sources, samples]}
     * @return a scalar, minus the mean best-permutation SI-SNR
     */
    public static Var loss(Var estimates, Var references) {
        int b = estimates.shape().dim(0);
        int n = estimates.shape().dim(1);
        int s = estimates.shape().dim(2);
        Var[][] pair = new Var[n][n];
        for (int i = 0; i < n; i++) {
            Var est = estimates.slice(1, i, i + 1).reshape(b, s);
            for (int j = 0; j < n; j++) {
                pair[i][j] = siSnr(est, references.slice(1, j, j + 1).reshape(b, s));
            }
        }
        List<Var> scores = new ArrayList<>();
        for (int[] p : permutations(n)) {
            Var sum = null;
            for (int j = 0; j < n; j++) {
                Var v = pair[p[j]][j];
                sum = sum == null ? v : sum.add(v);
            }
            scores.add(sum.mul(1f / n).reshape(b, 1));
        }
        Var all = Var.concat(1, scores.toArray(Var[]::new));
        return all.max(1, false).mean().neg();
    }

    /**
     * The best mean SI-SNR improvement of estimates over the mixture, under the best
     * assignment.
     *
     * @param estimates per output its samples
     * @param references per source its samples
     * @param mixture the input mixture
     * @return mean over sources of {@code SI-SNR(est, ref) - SI-SNR(mixture, ref)}, in dB
     */
    public static double improvement(float[][] estimates, float[][] references, float[] mixture) {
        int n = references.length;
        double best = Double.NEGATIVE_INFINITY;
        for (int[] p : permutations(n)) {
            double s = 0;
            for (int j = 0; j < n; j++) {
                s += siSnr(estimates[p[j]], references[j]) - siSnr(mixture, references[j]);
            }
            best = Math.max(best, s / n);
        }
        return best;
    }
}
