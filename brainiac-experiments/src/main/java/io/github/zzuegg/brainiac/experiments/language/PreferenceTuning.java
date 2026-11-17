package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Preference optimization on the counting language: the pretrained model continues
 * {@code a, a+1} upwards, the "human" prefers counting back down. Each objective (IPO, KTO,
 * ORPO, SimPO) tunes a copy of the model on preference pairs and the policy's margin
 * {@code log pi(chosen) - log pi(rejected)} is compared before and after.
 *
 * <p>{@link #noisy} compares the DPO family (DPO, cDPO, rDPO, DPOP, beta-DPO) when a share of
 * the preference labels is flipped: held-out win rate and the drift of the chosen answers'
 * likelihood.
 */
public final class PreferenceTuning {

    /** The objectives compared. */
    public enum Objective {
        /** identity preference optimization, against a frozen reference */
        IPO,
        /** Kahneman-Tversky optimization from unpaired good/bad labels */
        KTO,
        /** odds-ratio preference optimization, reference-free with an SFT term */
        ORPO,
        /** simple preference optimization, length-normalized and reference-free */
        SIMPO
    }

    /**
     * @param objective the objective
     * @param marginBefore the mean log-probability margin before
     * @param marginAfter the margin after
     */
    public record Result(Objective objective, double marginBefore, double marginAfter) {
    }

    private PreferenceTuning() {
    }

    private static final int PROMPT = 2;
    private static final int ANSWER = 4;

    /** {@code [2][batch][steps]}: chosen (counting down) and rejected (counting up). */
    static int[][][] pairs(SplittableRandom r, int batch, int vocab) {
        int[][] chosen = new int[batch][PROMPT + ANSWER];
        int[][] rejected = new int[batch][PROMPT + ANSWER];
        for (int b = 0; b < batch; b++) {
            int a = r.nextInt(vocab);
            for (int[] row : new int[][] {chosen[b], rejected[b]}) {
                row[0] = a;
                row[1] = (a + 1) % vocab;
            }
            for (int t = 0; t < ANSWER; t++) {
                chosen[b][PROMPT + t] = Math.floorMod(a - t, vocab);
                rejected[b][PROMPT + t] = (a + 2 + t) % vocab;
            }
        }
        return new int[][][] {chosen, rejected};
    }

    private static int[] from(int batch) {
        int[] f = new int[batch];
        java.util.Arrays.fill(f, PROMPT);
        return f;
    }

    /**
     * @param model the policy
     * @param seed the data seed
     * @return the mean summed log-probability margin of chosen over rejected
     */
    public static double margin(HookedDecoder model, long seed) {
        int[][][] p = pairs(new SplittableRandom(seed), 32, model.vocabSize());
        int[] f = from(32);
        return model.sequenceLogProb(p[0], f, false).sub(model.sequenceLogProb(p[1], f, false)).mean().value().item();
    }

    /**
     * Tunes a copy of {@code base} with one objective.
     *
     * @param base the pretrained model (not changed; also the reference)
     * @param objective the objective
     * @param steps the steps
     * @param seed the seed
     * @return the margins
     */
    public static Result run(HookedDecoder base, Objective objective, int steps, long seed) {
        HookedDecoder policy = base.copy();
        double before = margin(policy, seed + 50);
        Optimizer opt = Optimizer.adam(0.005f);
        SplittableRandom r = new SplittableRandom(seed);
        int batch = 16;
        int[] f = from(batch);
        for (int s = 0; s < steps; s++) {
            int[][][] p = pairs(r, batch, base.vocabSize());
            Var loss = switch (objective) {
                case IPO -> PreferenceLosses.ipo(policy.sequenceLogProb(p[0], f, false),
                        policy.sequenceLogProb(p[1], f, false), base.sequenceLogProb(p[0], f, false).detach(),
                        base.sequenceLogProb(p[1], f, false).detach(), 0.1f);
                case KTO -> {
                    int[][] mixed = new int[batch][];
                    boolean[] good = new boolean[batch];
                    for (int b = 0; b < batch; b++) {
                        good[b] = b % 2 == 0;
                        mixed[b] = good[b] ? p[0][b] : p[1][b];
                    }
                    int[][] mismatched = new int[batch][];
                    for (int b = 0; b < batch; b++) {
                        mismatched[b] = mixed[b].clone();
                        int[] other = mixed[(b + 1) % batch];
                        System.arraycopy(other, PROMPT, mismatched[b], PROMPT, ANSWER);
                    }
                    float z = PreferenceLosses.klEstimate(policy.sequenceLogProb(mismatched, f, false),
                            base.sequenceLogProb(mismatched, f, false));
                    yield PreferenceLosses.kto(policy.sequenceLogProb(mixed, f, false),
                            base.sequenceLogProb(mixed, f, false).detach(), good, 0.5f, 1, 1, z);
                }
                case ORPO -> {
                    Var meanChosen = policy.sequenceLogProb(p[0], f, true);
                    yield PreferenceLosses.orpo(meanChosen.mean().neg(), meanChosen,
                            policy.sequenceLogProb(p[1], f, true), 0.5f);
                }
                case SIMPO -> PreferenceLosses.simpo(policy.sequenceLogProb(p[0], f, true),
                        policy.sequenceLogProb(p[1], f, true), 2f, 0.5f);
            };
            loss.backward();
            opt.step(policy.parameters());
            policy.zeroGrad();
            base.zeroGrad();
        }
        return new Result(objective, before, margin(policy, seed + 50));
    }

    /** The DPO family compared under label noise. */
    public enum DpoVariant {
        /** plain DPO */
        DPO,
        /** conservative DPO: label smoothing at the flip rate */
        CDPO,
        /** robust DPO: unbiased noisy-label correction at the flip rate */
        RDPO,
        /** DPO-Positive: penalizes a falling chosen likelihood */
        DPOP,
        /** beta-DPO: batch-adaptive beta */
        BETA_DPO
    }

    /**
     * @param variant the loss
     * @param winRate held-out pairs whose chosen answer the tuned policy finds more likely
     *     than the rejected one
     * @param chosenDrift the change of the mean held-out {@code log pi(chosen)}
     * @param marginBefore the mean held-out log-probability margin before tuning
     * @param marginAfter the mean held-out log-probability margin after tuning
     */
    public record NoisyResult(DpoVariant variant, double winRate, double chosenDrift, double marginBefore,
            double marginAfter) {
    }

    private static double winRate(HookedDecoder model, long seed) {
        int[][][] p = pairs(new SplittableRandom(seed), 64, model.vocabSize());
        int[] f = from(64);
        float[] c = model.sequenceLogProb(p[0], f, false).value().toArray();
        float[] r = model.sequenceLogProb(p[1], f, false).value().toArray();
        int wins = 0;
        for (int i = 0; i < c.length; i++) {
            wins += c[i] > r[i] ? 1 : 0;
        }
        return (double) wins / c.length;
    }

    private static double chosenLogProb(HookedDecoder model, long seed) {
        int[][][] p = pairs(new SplittableRandom(seed), 64, model.vocabSize());
        return model.sequenceLogProb(p[0], from(64), false).mean().value().item();
    }

    /**
     * Tunes a copy of {@code base} on preference pairs whose labels are flipped at random
     * with probability {@code flipRate}, and measures the held-out (clean) win rate and how
     * far the chosen answers' likelihood moved.
     *
     * @param base the pretrained model (not changed; also the reference)
     * @param variant the loss
     * @param flipRate the share of flipped labels, also the epsilon given to cDPO and rDPO
     * @param steps the steps
     * @param seed the seed
     * @return the result
     */
    public static NoisyResult noisy(HookedDecoder base, DpoVariant variant, double flipRate, int steps, long seed) {
        HookedDecoder policy = base.copy();
        double chosenBefore = chosenLogProb(policy, seed + 50);
        double marginBefore = margin(policy, seed + 50);
        Optimizer opt = Optimizer.adam(0.005f);
        SplittableRandom r = new SplittableRandom(seed);
        int batch = 16;
        int[] f = from(batch);
        float beta = 0.5f;
        float eps = (float) flipRate;
        float m0 = 0;
        for (int s = 0; s < steps; s++) {
            int[][][] p = pairs(r, batch, base.vocabSize());
            for (int b = 0; b < batch; b++) {
                if (r.nextDouble() < flipRate) {
                    int[] t = p[0][b];
                    p[0][b] = p[1][b];
                    p[1][b] = t;
                }
            }
            Var pc = policy.sequenceLogProb(p[0], f, false);
            Var pr = policy.sequenceLogProb(p[1], f, false);
            Var rc = base.sequenceLogProb(p[0], f, false).detach();
            Var rr = base.sequenceLogProb(p[1], f, false).detach();
            Var loss = switch (variant) {
                case DPO -> PreferenceLosses.dpo(pc, pr, rc, rr, beta);
                case CDPO -> PreferenceLosses.conservativeDpo(pc, pr, rc, rr, beta, eps);
                case RDPO -> PreferenceLosses.robustDpo(pc, pr, rc, rr, beta, eps);
                case DPOP -> PreferenceLosses.dpoPositive(pc, pr, rc, rr, beta, 5f);
                case BETA_DPO -> {
                    float mean = PreferenceLosses.margin(pc, pr, rc, rr).mean().value().item();
                    Var l = PreferenceLosses.betaDpo(pc, pr, rc, rr, beta, 0.6f, m0);
                    m0 = s == 0 ? mean : 0.9f * m0 + 0.1f * mean;
                    yield l;
                }
            };
            loss.backward();
            opt.step(policy.parameters());
            policy.zeroGrad();
            base.zeroGrad();
        }
        return new NoisyResult(variant, winRate(policy, seed + 50), chosenLogProb(policy, seed + 50) - chosenBefore,
                marginBefore, margin(policy, seed + 50));
    }
}
