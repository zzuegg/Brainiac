package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Budget forcing (s1): control how long a reasoning model thinks at decode time, with no
 * training. Thinking is everything the model writes after the prompt until its
 * end-of-thinking token. Below the minimum budget the end-of-thinking token is suppressed, so
 * the model keeps going; when it would have ended, a {@code waitToken} (s1 appends "Wait") can
 * be forced in its place, nudging it to re-check. At the maximum budget the end-of-thinking
 * token is forced, and the model must answer with what it has. Sweeping the budget traces
 * accuracy against test-time compute.
 *
 * <p>{@link #processor} is a plain {@link LogitsProcessor} that reads the budget off the
 * context, so it drops into {@link DecodingPipeline} or, through
 * {@link ProcessedLanguageModel}, into the core {@code Generator}; {@link #generate} is the
 * driver that splits the output into thinking and answer.
 *
 * <p>Muennighoff et al., "s1: Simple test-time scaling" (2025),
 * <a href="https://arxiv.org/abs/2501.19393">arXiv:2501.19393</a>.
 */
public final class BudgetForcing {

    /**
     * A generation split at the end of thinking.
     *
     * @param thinking the thinking tokens
     * @param answer the tokens after the end-of-thinking token
     * @param ended whether thinking ended at all
     */
    public record Trace(int[] thinking, int[] answer, boolean ended) {
    }

    private BudgetForcing() {
    }

    /**
     * The budget as a processor.
     *
     * @param promptLength the prompt's length (thinking starts after it)
     * @param endOfThinking the end-of-thinking token
     * @param minThinking the fewest thinking tokens (0 for none)
     * @param maxThinking the most thinking tokens
     * @param waitToken a token forced instead of an early end, or -1 to only suppress the end
     * @return the processor
     */
    public static LogitsProcessor processor(int promptLength, int endOfThinking, int minThinking, int maxThinking,
            int waitToken) {
        if (minThinking < 0 || maxThinking < minThinking) {
            throw new IllegalArgumentException("need 0 <= min <= max, got " + minThinking + ", " + maxThinking);
        }
        return (context, logits) -> {
            int thought = 0;
            for (int i = promptLength; i < context.length; i++) {
                if (context[i] == endOfThinking) {
                    return;   // answering: no budget applies
                }
                thought++;
            }
            if (thought >= maxThinking) {
                only(logits, endOfThinking);
            } else if (thought < minThinking) {
                if (waitToken >= 0 && LogitMath.argmax(logits) == endOfThinking) {
                    only(logits, waitToken);
                } else {
                    logits[endOfThinking] = LogitMath.MASKED;
                }
            }
        };
    }

    private static void only(float[] logits, int token) {
        float keep = logits[token];
        Arrays.fill(logits, LogitMath.MASKED);
        logits[token] = Float.isFinite(keep) ? keep : 0;
    }

    /**
     * Thinks within a budget, then answers.
     *
     * @param model the model
     * @param prompt the question
     * @param endOfThinking the end-of-thinking token
     * @param minThinking the fewest thinking tokens
     * @param maxThinking the most thinking tokens
     * @param answerTokens tokens of answer to read after thinking
     * @param sampler how tokens are picked
     * @param random the source of randomness
     * @return the trace
     */
    public static Trace generate(LanguageModel model, int[] prompt, int endOfThinking, int minThinking,
            int maxThinking, int answerTokens, TokenSampler sampler, RandomGenerator random) {
        int[] out = DecodingPipeline.of(model)
                .processor(processor(prompt.length, endOfThinking, minThinking, maxThinking, -1))
                .sampler(sampler).generate(prompt, maxThinking + 1 + answerTokens, random);
        for (int i = 0; i < out.length; i++) {
            if (out[i] == endOfThinking) {
                return new Trace(Arrays.copyOf(out, i), Arrays.copyOfRange(out, i + 1, Math.min(out.length,
                        i + 1 + answerTokens)), true);
            }
        }
        return new Trace(out, new int[0], false);
    }
}
