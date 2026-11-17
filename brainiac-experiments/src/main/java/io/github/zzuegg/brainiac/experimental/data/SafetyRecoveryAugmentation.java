package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Safety recovery augmentation, the data half of "deep safety alignment": refusal examples
 * whose response <em>starts</em> harmful and then recovers. A shallowly aligned model
 * refuses only in its first tokens, so an attacker who prefills (or fine-tunes in) a few
 * compliant tokens gets the harmful continuation; training on
 * {@code prompt + harmful[0..k) -> refusal} for random {@code k} teaches the model to
 * return to a refusal from anywhere inside a harmful answer. The loss is put on the
 * refusal only, so the model never learns to produce the harmful prefix itself.
 *
 * <p>Qi et al. 2025, "Safety Alignment Should Be Made More Than Just a Few Tokens Deep",
 * <a href="https://arxiv.org/abs/2406.05946">arXiv:2406.05946</a>.
 */
public final class SafetyRecoveryAugmentation {

    /**
     * One training example: the conditioning tokens and the tokens to learn.
     *
     * @param prompt the prompt followed by a harmful prefix
     * @param completion the recovery (refusal) to learn
     */
    public record Example(int[] prompt, int[] completion) {
    }

    private SafetyRecoveryAugmentation() {
    }

    /**
     * Makes one recovery example.
     *
     * @param prompt a harmful prompt
     * @param harmful a harmful response to it
     * @param refusal the refusal to recover into
     * @param prefix how many harmful tokens precede the refusal
     * @return the example
     */
    public static Example recovery(int[] prompt, int[] harmful, int[] refusal, int prefix) {
        int k = Math.min(prefix, harmful.length);
        int[] p = Arrays.copyOf(prompt, prompt.length + k);
        System.arraycopy(harmful, 0, p, prompt.length, k);
        return new Example(p, refusal.clone());
    }

    /**
     * Augments a refusal data set: each pair yields its plain refusal example and, with
     * probability {@code rate}, one recovery example with a prefix of {@code 1..maxPrefix}
     * harmful tokens.
     *
     * @param prompts harmful prompts
     * @param harmful a harmful response per prompt
     * @param refusal the refusal
     * @param rate the share of prompts that also get a recovery example
     * @param maxPrefix the longest harmful prefix
     * @param random the random source
     * @return the examples
     */
    public static List<Example> augment(List<int[]> prompts, List<int[]> harmful, int[] refusal, double rate,
            int maxPrefix, SplittableRandom random) {
        List<Example> out = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i++) {
            out.add(new Example(prompts.get(i).clone(), refusal.clone()));
            if (random.nextDouble() < rate) {
                int k = 1 + random.nextInt(Math.max(1, Math.min(maxPrefix, harmful.get(i).length)));
                out.add(recovery(prompts.get(i), harmful.get(i), refusal, k));
            }
        }
        return out;
    }
}
