package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;

import java.util.Arrays;
import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * A decoding loop over any {@link LanguageModel} with the pieces the core
 * {@code Generator} fixes made pluggable: a chain of {@link LogitsProcessor}s, a
 * {@link TokenSampler} (greedy, ancestral, {@link Mirostat}, ...) and an optional
 * {@link TokenConstraint} for structured output. One token per model step, with the
 * model's cache.
 *
 * <pre>{@code
 * int[] out = DecodingPipeline.of(model)
 *         .processor(LogitsProcessors.dry(0.8f, 1.75f, 2, Set.of()))
 *         .processor(LogitsProcessors.temperature(0.8f))
 *         .processor(LogitsProcessors.typical(0.9))
 *         .sampler(Mirostat.v2(5, 0.1))
 *         .generate(prompt, 100, new SplittableRandom(1));
 * }</pre>
 */
public final class DecodingPipeline {

    private final LanguageModel model;
    private LogitsProcessor processor = LogitsProcessor.identity();
    private TokenSampler sampler = TokenSampler.greedy();
    private TokenConstraint constraint;
    private int eos = -1;

    private DecodingPipeline(LanguageModel model) {
        this.model = Objects.requireNonNull(model);
    }

    /**
     * @param model the model
     * @return a greedy pipeline without processors
     */
    public static DecodingPipeline of(LanguageModel model) {
        return new DecodingPipeline(model);
    }

    /**
     * @param next a processor, run after those added before
     * @return this pipeline
     */
    public DecodingPipeline processor(LogitsProcessor next) {
        processor = processor.andThen(next);
        return this;
    }

    /**
     * @param s the sampler
     * @return this pipeline
     */
    public DecodingPipeline sampler(TokenSampler s) {
        this.sampler = Objects.requireNonNull(s);
        return this;
    }

    /**
     * @param c the constraint the output must satisfy, or null
     * @return this pipeline
     */
    public DecodingPipeline constraint(TokenConstraint c) {
        this.constraint = c;
        return this;
    }

    /**
     * @param token the token that ends a sequence (not included in the output), or -1
     * @return this pipeline
     */
    public DecodingPipeline eos(int token) {
        this.eos = token;
        return this;
    }

    /**
     * Generates a continuation.
     *
     * @param prompt the prompt, at least one token
     * @param maxNew the most tokens to add
     * @param random the source of randomness
     * @return the new tokens (without the prompt and without {@code eos})
     */
    public int[] generate(int[] prompt, int maxNew, RandomGenerator random) {
        sampler.reset();
        LanguageModel.State state = model.newState(1);
        float[] logits = model.next(state, new int[][] {prompt}, null).toArray();
        int[] context = prompt.clone();
        int cs = constraint == null ? 0 : constraint.start();
        int n = 0;
        int[] out = new int[maxNew];
        while (n < maxNew) {
            float[] l = logits.clone();
            processor.process(context, l);
            if (constraint != null) {
                constraint.mask(cs, l);
            }
            int token = sampler.sample(context, l, random);
            if (token == eos) {
                break;
            }
            if (constraint != null) {
                cs = constraint.advance(cs, token);
            }
            out[n++] = token;
            context = Arrays.copyOf(context, context.length + 1);
            context[context.length - 1] = token;
            if (n < maxNew) {
                logits = model.next(state, new int[][] {{token}}, null).toArray();
            }
        }
        return Arrays.copyOf(out, n);
    }
}
