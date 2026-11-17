package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.FimTransform;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * The "FIM is free" check (Bavarian et al., <a href="https://arxiv.org/abs/2207.14255">2207.14255</a>)
 * on a toy language whose middle can only be known from its end. A document is three blocks
 * of two symbols, {@code u v w}, where {@code u} and {@code w} are random and {@code v} is a
 * copy of {@code w}. Each token spells its position and its symbol (id
 * {@code position * SYMBOLS + symbol}), as real text carries its own context, so a model can
 * tell where a token belongs without counting. Read left to right, the last block is
 * predictable (it repeats the middle); asked to fill in {@code v} between {@code u} and
 * {@code w}, only a model that can look at the suffix succeeds.
 *
 * <p>Two {@link HookedDecoder}s train for the same number of steps: one on plain documents,
 * one with half of its batches rearranged by {@link FimTransform} (PSM, random cut points).
 * Both are scored on infilling the middle block (the plain model can only continue the
 * prefix) and on the ordinary left-to-right loss of plain documents.
 */
@SuppressWarnings("exports")
public final class FillInTheMiddle {
    /** Symbols in the toy language. */
    public static final int SYMBOLS = 8;
    private static final int BLOCK = 2;
    private static final int TOKENS = 3 * BLOCK * SYMBOLS;
    private final FimTransform fim = FimTransform.of(TOKENS, TOKENS + 1, TOKENS + 2, FimTransform.Mode.PSM);

    /**
     * The outcome for one model.
     *
     * @param infillTokenAccuracy the share of middle tokens filled in correctly
     * @param infillExact the share of middles filled in entirely correctly
     * @param leftToRightLoss the next-token loss on plain documents
     */
    public record Score(double infillTokenAccuracy, double infillExact, double leftToRightLoss) {
    }

    private FillInTheMiddle() {
    }

    /**
     * Creates the experiment.
     *
     * @return the experiment
     */
    public static FillInTheMiddle create() {
        return new FillInTheMiddle();
    }

    /**
     * Draws a document.
     *
     * @param random the random source
     * @return {@code u v w} with {@code v = w}, as position-and-symbol token ids
     */
    public static int[] document(SplittableRandom random) {
        int[] d = new int[3 * BLOCK];
        for (int i = 0; i < BLOCK; i++) {
            d[i] = random.nextInt(SYMBOLS);
            d[2 * BLOCK + i] = random.nextInt(SYMBOLS);
            d[BLOCK + i] = d[2 * BLOCK + i];
        }
        for (int i = 0; i < d.length; i++) {
            d[i] += i * SYMBOLS;
        }
        return d;
    }

    /**
     * Trains a model.
     *
     * @param fimRate the share of batches rearranged for infilling, 0 for a plain model
     * @param steps optimizer steps of 32 documents
     * @param seed the seed
     * @return the model
     */
    public HookedDecoder train(double fimRate, int steps, long seed) {
        HookedDecoder model = HookedDecoder.create(HookedDecoder.Settings.tiny(TOKENS + 3), seed);
        Optimizer opt = Optimizer.adam(1e-2f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            boolean transform = r.nextDouble() < fimRate;
            int[][] batch = new int[32][];
            for (int i = 0; i < batch.length; i++) {
                int[] d = document(r);
                batch[i] = transform ? fim.apply(d, r) : d;
            }
            Var loss = model.loss(batch, true);
            loss.backward();
            opt.step(model.parameters());
            model.zeroGrad();
        }
        return model;
    }

    private static int argmax(float[] logits, int offset, int limit) {
        int best = 0;
        for (int v = 1; v < limit; v++) {
            if (logits[offset + v] > logits[offset + best]) {
                best = v;
            }
        }
        return best;
    }

    private static int[][] greedy(HookedDecoder model, int[][] prompts, int tokens) {
        int n = prompts.length;
        int vocab = model.vocabSize();
        int[][] seq = new int[n][];
        for (int i = 0; i < n; i++) {
            seq[i] = prompts[i].clone();
        }
        int[][] out = new int[n][tokens];
        for (int t = 0; t < tokens; t++) {
            float[] logits = model.forward(seq).logits().value().toArray();
            int len = seq[0].length;
            for (int i = 0; i < n; i++) {
                out[i][t] = argmax(logits, ((i + 1) * len - 1) * vocab, TOKENS);
                seq[i] = java.util.Arrays.copyOf(seq[i], len + 1);
                seq[i][len] = out[i][t];
            }
        }
        return out;
    }

    /**
     * Scores a model.
     *
     * @param model the model
     * @param useSuffix whether to infill with the FIM prompt (else continue the prefix)
     * @param seed the evaluation seed
     * @return the score
     */
    public Score evaluate(HookedDecoder model, boolean useSuffix, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = 128;
        int[][] docs = new int[n][];
        int[][] prompts = new int[n][];
        for (int i = 0; i < n; i++) {
            docs[i] = document(r);
            int[] prefix = java.util.Arrays.copyOfRange(docs[i], 0, BLOCK);
            int[] suffix = java.util.Arrays.copyOfRange(docs[i], 2 * BLOCK, 3 * BLOCK);
            prompts[i] = useSuffix ? fim.infillPrompt(prefix, suffix) : prefix;
        }
        int[][] middles = greedy(model, prompts, BLOCK);
        int right = 0;
        int exact = 0;
        for (int i = 0; i < n; i++) {
            int ok = 0;
            for (int t = 0; t < BLOCK; t++) {
                ok += middles[i][t] == docs[i][BLOCK + t] ? 1 : 0;
            }
            right += ok;
            exact += ok == BLOCK ? 1 : 0;
        }
        double l2r = model.loss(docs, false).value().item();
        return new Score((double) right / (n * BLOCK), (double) exact / n, l2r);
    }
}
