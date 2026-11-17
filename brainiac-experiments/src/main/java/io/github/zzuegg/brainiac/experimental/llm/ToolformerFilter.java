package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Toolformer's self-supervised filter (<a href="https://arxiv.org/abs/2302.04761">2302.04761</a>):
 * a candidate tool call inserted into a text is kept only if seeing the call and its
 * result makes the rest of the text more likely. With {@code L+} the loss of the
 * continuation after {@code call + result} and {@code L-} the smaller of the loss with no
 * call and with the call but an empty result, a call is useful when
 * {@code L- - L+ >= tau}. The kept calls become fine-tuning data that teaches the model
 * when and how to call the tool, without any human annotation.
 */
public final class ToolformerFilter {

    /**
     * A candidate call.
     *
     * @param prompt the text before the call
     * @param call the call tokens, from the open bracket up to and including the result
     *     marker
     * @param result the tool's result tokens, closing bracket included
     * @param emptyResult just the closing bracket, the "call without result" control
     * @param continuation the text after the call, whose loss decides
     */
    public record Candidate(int[] prompt, int[] call, int[] result, int[] emptyResult, int[] continuation) {

        /** @return the augmented completion: call, result, continuation */
        public int[] augmented() {
            return concat(concat(call, result), continuation);
        }
    }

    private final WindowLm lm;
    private final double threshold;

    private ToolformerFilter(WindowLm lm, double threshold) {
        this.lm = Objects.requireNonNull(lm);
        this.threshold = threshold;
    }

    /**
     * @param lm the model whose losses decide
     * @param threshold the loss reduction a call must achieve
     * @return the filter
     */
    public static ToolformerFilter of(WindowLm lm, double threshold) {
        return new ToolformerFilter(lm, threshold);
    }

    static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /**
     * @param c a candidate
     * @return {@code L- - L+}: how much the call lowers the continuation's loss (in nats)
     */
    public double gain(Candidate c) {
        double without = -lm.sequenceLogProb(c.prompt(), c.continuation());
        double emptyCall = -lm.sequenceLogProb(concat(concat(c.prompt(), c.call()), c.emptyResult()),
                c.continuation());
        double with = -lm.sequenceLogProb(concat(concat(c.prompt(), c.call()), c.result()), c.continuation());
        return Math.min(without, emptyCall) - with;
    }

    /**
     * @param candidates the candidates
     * @return those whose gain reaches the threshold
     */
    public List<Candidate> filter(List<Candidate> candidates) {
        List<Candidate> kept = new ArrayList<>();
        for (Candidate c : candidates) {
            if (gain(c) >= threshold) {
                kept.add(c);
            }
        }
        return kept;
    }

    /**
     * Like {@link #filter}, but of several candidate calls at the same place (same prompt
     * and continuation) keeps only the one with the largest gain, as Toolformer keeps the
     * best call per position.
     *
     * @param candidates the candidates
     * @return the best candidate per place, if it reaches the threshold
     */
    public List<Candidate> filterBestPerPlace(List<Candidate> candidates) {
        java.util.Map<String, Candidate> best = new java.util.LinkedHashMap<>();
        java.util.Map<String, Double> bestGain = new java.util.HashMap<>();
        for (Candidate c : candidates) {
            double g = gain(c);
            if (g < threshold) {
                continue;
            }
            String key = Arrays.toString(c.prompt()) + Arrays.toString(c.continuation());
            if (g > bestGain.getOrDefault(key, Double.NEGATIVE_INFINITY)) {
                bestGain.put(key, g);
                best.put(key, c);
            }
        }
        return new ArrayList<>(best.values());
    }
}
