package io.github.zzuegg.brainiac.experimental.serving;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * SpecTr: lossless verification of several independent draft candidates (Sun et al., NeurIPS
 * 2023, <a href="https://arxiv.org/abs/2310.15141">arXiv:2310.15141</a>). With {@code k} drafts
 * drawn i.i.d. from the draft distribution {@code q}, the best lossless rule is an optimal
 * transport coupling between {@code q^k} and the target {@code p}; SpecTr's k-sequential
 * selection (K-SEQ, Algorithm 2) is a cheap rule within {@code 1 - 1/e} of it:
 * <ol>
 * <li>{@code beta(rho) = sum_x min(q(x), p(x) / rho)} and {@code rho*} solves
 *     {@code 1 - (1 - beta(rho))^k = rho beta(rho)} on {@code [1, k]} (binary search);</li>
 * <li>scan the drafts and accept draft {@code i} with probability
 *     {@code min(1, p(x_i) / (rho q(x_i)))}; return the first accepted;</li>
 * <li>if none is accepted, sample the residual
 *     {@code (p(x) - min(q(x), p(x)/rho) (1 - (1 - beta)^k) / beta) / (1 - p_acc)}.</li>
 * </ol>
 * The output is distributed exactly as {@code p}. With {@code k = 1} and {@code rho = 1} this is
 * the usual speculative sampling step. {@link #naive} is the biased shortcut (accept the first
 * draft that passes the single-draft test, otherwise sample the single-draft residual) kept to
 * show the bias.
 *
 * <p>{@link #generate} verifies whole draft sequences: at each position the drafts still
 * agreeing with the accepted prefix are i.i.d. samples of the draft's next-token distribution,
 * so K-SEQ applies among them; on a residual sample the step ends, and if every position is
 * accepted the target adds one bonus token.
 */
public final class MultiDraftVerifier {

    private MultiDraftVerifier() {
    }

    /**
     * One verified token.
     *
     * @param token the output token
     * @param acceptedDraft the index of the accepted draft, or -1 if the residual was sampled
     */
    public record Outcome(int token, int acceptedDraft) {
    }

    /**
     * {@code beta(rho)}.
     *
     * @param p the target
     * @param q the draft
     * @param rho the ratio
     * @return the overlap
     */
    public static double beta(double[] p, double[] q, double rho) {
        double b = 0;
        for (int x = 0; x < p.length; x++) {
            b += Math.min(q[x], p[x] / rho);
        }
        return b;
    }

    /**
     * Solves {@code 1 - (1 - beta(rho))^k = rho beta(rho)} by bisection on {@code [1, k]}.
     *
     * @param p the target
     * @param q the draft
     * @param k the drafts
     * @return {@code rho*}
     */
    public static double rho(double[] p, double[] q, int k) {
        double lo = 1;
        double hi = Math.max(1, k);
        for (int it = 0; it < 50; it++) {
            double mid = (lo + hi) / 2;
            double b = beta(p, q, mid);
            double f = 1 - Math.pow(1 - b, k) - mid * b;
            if (f > 0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return hi;
    }

    /**
     * K-SEQ verification of one position.
     *
     * @param p the target distribution
     * @param q the draft distribution the drafts were drawn from
     * @param drafts the draft tokens, i.i.d. from {@code q}
     * @param random the random source
     * @return the token
     */
    public static Outcome verify(double[] p, double[] q, int[] drafts, SplittableRandom random) {
        int k = drafts.length;
        if (k == 0) {
            return new Outcome(sample(p, random), -1);
        }
        double rho = rho(p, q, k);
        for (int i = 0; i < k; i++) {
            int x = drafts[i];
            double a = q[x] <= 0 ? 0 : Math.min(1, p[x] / (rho * q[x]));
            if (random.nextDouble() < a) {
                return new Outcome(x, i);
            }
        }
        double b = beta(p, q, rho);
        double pAcc = 1 - Math.pow(1 - b, k);
        double[] res = new double[p.length];
        double z = 0;
        for (int x = 0; x < p.length; x++) {
            res[x] = Math.max(0, p[x] - Math.min(q[x], p[x] / rho) * (b <= 0 ? 0 : pAcc / b));
            z += res[x];
        }
        if (z <= 1e-15) {
            return new Outcome(sample(p, random), -1);
        }
        for (int x = 0; x < p.length; x++) {
            res[x] /= z;
        }
        return new Outcome(sample(res, random), -1);
    }

    /**
     * The biased shortcut: each draft gets the single-draft test {@code min(1, p/q)} in turn,
     * and on rejection of all the single-draft residual {@code max(p - q, 0)} is sampled. It
     * over-weights tokens the draft likes.
     *
     * @param p the target
     * @param q the draft
     * @param drafts the drafts
     * @param random the random source
     * @return the token
     */
    public static Outcome naive(double[] p, double[] q, int[] drafts, SplittableRandom random) {
        for (int i = 0; i < drafts.length; i++) {
            int x = drafts[i];
            if (random.nextDouble() < Math.min(1, p[x] / Math.max(1e-300, q[x]))) {
                return new Outcome(x, i);
            }
        }
        double[] res = new double[p.length];
        double z = 0;
        for (int x = 0; x < p.length; x++) {
            res[x] = Math.max(0, p[x] - q[x]);
            z += res[x];
        }
        if (z <= 1e-15) {
            return new Outcome(sample(p, random), -1);
        }
        for (int x = 0; x < p.length; x++) {
            res[x] /= z;
        }
        return new Outcome(sample(res, random), -1);
    }

    /**
     * Speculative generation with {@code k} independent draft sequences of length
     * {@code length} per target pass.
     *
     * @param target the target's next-token distribution given a prefix
     * @param draft the draft's next-token distribution given a prefix
     * @param prompt the prompt
     * @param tokens tokens to generate
     * @param k drafts per pass
     * @param length draft length
     * @param random the random source
     * @return {@code {generated tokens, target passes}}
     */
    public static int[][] generate(Function<int[], double[]> target, Function<int[], double[]> draft, int[] prompt,
            int tokens, int k, int length, SplittableRandom random) {
        List<Integer> out = new ArrayList<>();
        int passes = 0;
        int[] seq = prompt.clone();
        while (out.size() < tokens) {
            passes++;
            int[][] drafts = new int[k][length];
            for (int i = 0; i < k; i++) {
                int[] s = seq.clone();
                for (int j = 0; j < length; j++) {
                    drafts[i][j] = sample(draft.apply(s), random);
                    s = append(s, drafts[i][j]);
                }
            }
            List<Integer> alive = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                alive.add(i);
            }
            boolean all = true;
            for (int j = 0; j < length && out.size() < tokens; j++) {
                double[] p = target.apply(seq);
                double[] q = draft.apply(seq);
                int[] cand = new int[alive.size()];
                for (int i = 0; i < cand.length; i++) {
                    cand[i] = drafts[alive.get(i)][j];
                }
                Outcome o = verify(p, q, cand, random);
                out.add(o.token());
                seq = append(seq, o.token());
                if (o.acceptedDraft() < 0) {
                    all = false;
                    break;
                }
                int pos = j;
                alive.removeIf(i -> drafts[i][pos] != o.token());
            }
            if (all && out.size() < tokens) {
                int bonus = sample(target.apply(seq), random);
                out.add(bonus);
                seq = append(seq, bonus);
            }
        }
        return new int[][] {out.stream().mapToInt(Integer::intValue).toArray(), {passes}};
    }

    private static int[] append(int[] s, int t) {
        int[] o = Arrays.copyOf(s, s.length + 1);
        o[s.length] = t;
        return o;
    }

    static int sample(double[] p, SplittableRandom random) {
        double u = random.nextDouble();
        double c = 0;
        for (int i = 0; i < p.length; i++) {
            c += p[i];
            if (u < c) {
                return i;
            }
        }
        return p.length - 1;
    }
}
