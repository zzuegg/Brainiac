package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.Sampling;

/**
 * Changes one row of next-token logits in place before a token is picked, given the tokens
 * so far: penalties, truncation (set a logit to {@link LogitMath#MASKED}), temperature,
 * biases, watermarks, grammar masks. Processors chain with {@link #andThen}, in the order
 * Hugging Face's {@code LogitsProcessorList} runs them.
 *
 * <p>The core {@code Generator} fixes its steps in {@link Sampling}; this is the hook it
 * lacks, and {@link #of(Sampling)} turns a core setting into a processor so both mix.
 */
@FunctionalInterface
public interface LogitsProcessor {

    /**
     * Changes the logits.
     *
     * @param context every token so far (prompt and generated)
     * @param logits the row of logits; changed in place
     */
    void process(int[] context, float[] logits);

    /**
     * @param next the processor to run after this one
     * @return both, this one first
     */
    default LogitsProcessor andThen(LogitsProcessor next) {
        return (c, l) -> {
            process(c, l);
            next.process(c, l);
        };
    }

    /** @return a processor that changes nothing */
    static LogitsProcessor identity() {
        return (c, l) -> { };
    }

    /**
     * The core sampling settings as a processor: repetition penalty, then the truncation of
     * temperature, top-k, top-p and min-p (cut tokens become {@link LogitMath#MASKED}, the
     * rest the log of their kept probability). Greedy settings keep only the argmax.
     *
     * @param sampling the settings
     * @return the processor
     */
    static LogitsProcessor of(Sampling sampling) {
        return (c, l) -> {
            sampling.penalize(l, c);
            double[] p = sampling.probabilities(l);
            for (int i = 0; i < l.length; i++) {
                l[i] = p[i] > 0 ? (float) Math.log(p[i]) : LogitMath.MASKED;
            }
        };
    }
}
