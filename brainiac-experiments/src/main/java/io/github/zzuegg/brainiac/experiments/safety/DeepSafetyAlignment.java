package io.github.zzuegg.brainiac.experiments.safety;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SafetyRecoveryAugmentation;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.train.TokenwiseConstrainedLoss;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Deep safety alignment on the {@link SafetyToyLm} refusal toy, in two halves.
 *
 * <ol>
 *   <li><b>Prefilling attacks.</b> A complying model is aligned with standard refusal SFT,
 *       or with SFT plus {@link SafetyRecoveryAugmentation} (harmful prefixes followed by a
 *       refusal). With one or two harmful tokens prefilled, does the model finish the
 *       harmful answer or recover into a refusal?</li>
 *   <li><b>Fine-tuning attacks.</b> Each aligned model is then fine-tuned on benign data
 *       whose answers start with an affirmative "Sure" mixed with one harmful example in
 *       eight (the paper's harmful fine-tuning attack), either with plain cross-entropy or
 *       with the
 *       {@link TokenwiseConstrainedLoss} that pins the first response token to the aligned
 *       model (beta 4 on the first token, 0.1 after). The outcome is the refusal rate on harmful prompts afterwards, with the
 *       benign task's accuracy as the cost.</li>
 * </ol>
 *
 * <p>Qi et al. 2025, "Safety Alignment Should Be Made More Than Just a Few Tokens Deep",
 * <a href="https://arxiv.org/abs/2406.05946">arXiv:2406.05946</a>.
 */
public final class DeepSafetyAlignment {

    /**
     * Prefilling results of one alignment.
     *
     * @param alignment "standard" or "augmented"
     * @param refusal harmful prompts refused
     * @param prefill1Harm harmful answers finished after one prefilled token
     * @param prefill2Harm harmful answers finished after two prefilled tokens
     * @param prefill1Recover refusals right after one prefilled token
     */
    public record Prefill(String alignment, double refusal, double prefill1Harm, double prefill2Harm,
            double prefill1Recover) {
    }

    /**
     * Fine-tuning attack results.
     *
     * @param alignment "standard" or "augmented"
     * @param objective "plain SFT" or "constrained"
     * @param refusalAfter harmful prompts refused after the fine-tuning
     * @param taskAccuracy benign prompts whose completion ends in the correct answer
     *     {@code x, x+1, END}, whatever opener precedes it
     */
    public record FineTune(String alignment, String objective, double refusalAfter, double taskAccuracy) {
    }

    /**
     * Everything.
     *
     * @param prefill one entry per alignment
     * @param fineTune one entry per alignment and objective
     */
    public record Outcome(List<Prefill> prefill, List<FineTune> fineTune) {
    }

    private DeepSafetyAlignment() {
    }

    /**
     * Runs the experiment.
     *
     * @param alignSteps alignment steps
     * @param attackSteps fine-tuning attack steps
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int alignSteps, int attackSteps, long seed) {
        WindowLm base = SafetyToyLm.train(SafetyToyLm.Behaviour.UNALIGNED, 500, seed);
        List<Prefill> prefill = new ArrayList<>();
        List<FineTune> fineTune = new ArrayList<>();
        for (boolean augmented : new boolean[] {false, true}) {
            String name = augmented ? "augmented" : "standard";
            WindowLm aligned = align(base.copy(), augmented, alignSteps, seed);
            SplittableRandom r = new SplittableRandom(seed + 99);
            List<int[]> harmful = SafetyToyLm.prompts(100, true, false, r);
            prefill.add(new Prefill(name, SafetyToyLm.rate(aligned, harmful, SafetyToyLm::refuses),
                    SafetyToyLm.rate(aligned, SafetyToyLm.prefilled(harmful, 1), SafetyToyLm.finishesHarm(1)),
                    SafetyToyLm.rate(aligned, SafetyToyLm.prefilled(harmful, 2), SafetyToyLm.finishesHarm(2)),
                    SafetyToyLm.rate(aligned, SafetyToyLm.prefilled(harmful, 1), SafetyToyLm::refuses)));
            for (boolean constrained : new boolean[] {false, true}) {
                WindowLm attacked = aligned.copy();
                attack(attacked, aligned, constrained, attackSteps, seed);
                List<int[]> benign = SafetyToyLm.prompts(100, false, false, r);
                int ok = 0;
                int[][] outs = attacked.sample(benign.toArray(new int[0][]), 5, SafetyToyLm.END, 0,
                        new SplittableRandom(0));
                for (int i = 0; i < outs.length; i++) {
                    ok += endsWith(outs[i], SafetyToyLm.benignResponse(benign.get(i))) ? 1 : 0;
                }
                fineTune.add(new FineTune(name, constrained ? "constrained" : "plain SFT",
                        SafetyToyLm.rate(attacked, harmful, SafetyToyLm::refuses), ok / (double) benign.size()));
            }
        }
        return new Outcome(prefill, fineTune);
    }

    private static boolean endsWith(int[] out, int[] answer) {
        return out.length >= answer.length
                && Arrays.equals(Arrays.copyOfRange(out, out.length - answer.length, out.length), answer);
    }

    private static WindowLm align(WindowLm lm, boolean augmented, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 5);
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int s = 0; s < steps; s++) {
            List<int[]> p = new ArrayList<>();
            List<int[]> c = new ArrayList<>();
            for (int[] b : SafetyToyLm.plainPrompts(16, false, r)) {
                p.add(b);
                c.add(SafetyToyLm.benignResponse(b));
            }
            List<int[]> harmful = SafetyToyLm.plainPrompts(16, true, r);
            if (augmented) {
                List<int[]> answers = new ArrayList<>();
                for (int i = 0; i < harmful.size(); i++) {
                    answers.add(SafetyToyLm.HARMFUL_RESPONSE);
                }
                for (SafetyRecoveryAugmentation.Example e : SafetyRecoveryAugmentation.augment(harmful, answers,
                        SafetyToyLm.REFUSAL, 0.5, 3, r)) {
                    p.add(e.prompt());
                    c.add(e.completion());
                }
            } else {
                for (int[] h : harmful) {
                    p.add(h);
                    c.add(SafetyToyLm.REFUSAL);
                }
            }
            lm.supervisedStep(p, c, opt);
        }
        return lm;
    }

    private static void attack(WindowLm lm, WindowLm aligned, boolean constrained, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 11);
        Optimizer opt = Optimizer.adam(5e-3f);
        for (int s = 0; s < steps; s++) {
            List<int[]> p = SafetyToyLm.plainPrompts(28, false, r);
            List<int[]> c = new ArrayList<>();
            for (int[] b : p) {
                c.add(SafetyToyLm.sureResponse(b));
            }
            for (int[] h : SafetyToyLm.plainPrompts(4, true, r)) {
                p.add(h);
                c.add(SafetyToyLm.HARMFUL_RESPONSE);
            }
            if (!constrained) {
                lm.supervisedStep(p, c, opt);
                continue;
            }
            Var logp = lm.logProbs(p, c);
            int[] pos = new int[(int) logp.shape().size()];
            int k = 0;
            for (int[] comp : c) {
                for (int t = 0; t < comp.length; t++) {
                    pos[k++] = t;
                }
            }
            Var loss = TokenwiseConstrainedLoss.loss(logp, aligned.logProbs(p, c).value(),
                    TokenwiseConstrainedLoss.schedule(pos, 1, 4f, 0.1f));
            loss.backward();
            opt.step(lm.parameters());
        }
    }
}
