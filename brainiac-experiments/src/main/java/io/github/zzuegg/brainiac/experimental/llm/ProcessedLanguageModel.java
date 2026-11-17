package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * A {@link LanguageModel} whose logits pass through a {@link LogitsProcessor}, so the core
 * {@code Generator}, which has no processor hook, runs any processor unchanged: wrap the
 * model and keep the generator. The wrapper remembers each row's tokens to give the
 * processor its context (padding included as read). Masked tokens come out as a very
 * negative finite logit, which the core sampling treats as probability zero.
 */
public final class ProcessedLanguageModel implements LanguageModel {

    private static final float FLOOR = -1e30f;

    private final LanguageModel model;
    private final LogitsProcessor processor;

    /**
     * @param model the model
     * @param processor the processor applied to every row of every prediction
     */
    public ProcessedLanguageModel(LanguageModel model, LogitsProcessor processor) {
        this.model = model;
        this.processor = processor;
    }

    private final class Wrapped implements State {
        private State inner;
        private int[][] tokens;

        Wrapped(State inner, int batch) {
            this.inner = inner;
            this.tokens = new int[batch][0];
        }

        @Override
        public int length() {
            return inner.length();
        }

        @Override
        public State select(int[] rows) {
            inner = inner.select(rows);
            int[][] t = new int[rows.length][];
            for (int i = 0; i < rows.length; i++) {
                t[i] = tokens[rows[i]].clone();
            }
            tokens = t;
            return this;
        }

        @Override
        public State truncate(int length) {
            inner = inner.truncate(length);
            for (int b = 0; b < tokens.length; b++) {
                tokens[b] = Arrays.copyOf(tokens[b], length);
            }
            return this;
        }
    }

    @Override
    public int vocabSize() {
        return model.vocabSize();
    }

    @Override
    public State newState(int batch) {
        return new Wrapped(model.newState(batch), batch);
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        Wrapped w = (Wrapped) state;
        Tensor logits = model.next(w.inner, tokens, padding);
        append(w, tokens);
        int batch = tokens.length;
        int vocab = vocabSize();
        float[] all = logits.toArray();
        for (int b = 0; b < batch; b++) {
            float[] row = Arrays.copyOfRange(all, b * vocab, (b + 1) * vocab);
            processor.process(w.tokens[b], row);
            for (int i = 0; i < vocab; i++) {
                all[b * vocab + i] = Float.isFinite(row[i]) ? row[i] : FLOOR;
            }
        }
        return Tensor.of(Shape.of(batch, vocab), all);
    }

    @Override
    public Tensor nextAll(State state, int[][] tokens) {
        Wrapped w = (Wrapped) state;
        Tensor logits = model.nextAll(w.inner, tokens);
        int batch = tokens.length;
        int steps = tokens[0].length;
        int vocab = vocabSize();
        float[] all = logits.toArray();
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < steps; s++) {
                int[] context = Arrays.copyOf(w.tokens[b], w.tokens[b].length + s + 1);
                System.arraycopy(tokens[b], 0, context, w.tokens[b].length, s + 1);
                int off = (b * steps + s) * vocab;
                float[] row = Arrays.copyOfRange(all, off, off + vocab);
                processor.process(context, row);
                for (int i = 0; i < vocab; i++) {
                    all[off + i] = Float.isFinite(row[i]) ? row[i] : FLOOR;
                }
            }
        }
        append(w, tokens);
        return Tensor.of(Shape.of(batch, steps, vocab), all);
    }

    private static void append(Wrapped w, int[][] tokens) {
        for (int b = 0; b < tokens.length; b++) {
            int[] t = Arrays.copyOf(w.tokens[b], w.tokens[b].length + tokens[b].length);
            System.arraycopy(tokens[b], 0, t, w.tokens[b].length, tokens[b].length);
            w.tokens[b] = t;
        }
    }
}
