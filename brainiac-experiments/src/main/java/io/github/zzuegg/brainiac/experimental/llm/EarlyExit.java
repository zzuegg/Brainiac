package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Spending fewer layers on easy tokens, for a {@link HookedDecoder} whose output head can read
 * any layer:
 * <ul>
 * <li>{@link #deepSupervisionLoss}: trains every layer's exit through the shared head
 *     (a weighted sum of per-layer cross-entropies, as CALM and LayerSkip do), so
 *     intermediate layers make usable predictions;</li>
 * <li>{@link #generate}: CALM-style confident adaptive decoding (Schuster et al. 2022,
 *     <a href="https://arxiv.org/abs/2207.07061">arXiv:2207.07061</a>): run blocks one at a
 *     time and stop as soon as the softmax confidence of the next token passes a
 *     threshold. (The context is recomputed with every block, as the decoder's state does, so
 *     there is no need for CALM's state copying.)</li>
 * <li>{@link #selfSpeculative}: self-speculative decoding (Draft and Verify, Zhang et al.
 *     2023, <a href="https://arxiv.org/abs/2309.08168">arXiv:2309.08168</a>; LayerSkip,
 *     Elhoushi et al. 2024, <a href="https://arxiv.org/abs/2404.16710">arXiv:2404.16710</a>):
 *     the first blocks with an early exit draft tokens, the full model verifies them in one
 *     pass; greedy output is exactly the full model's.</li>
 * </ul>
 * Skipping arbitrary blocks at inference is {@link HookedDecoder#skip}.
 */
public final class EarlyExit {

    private EarlyExit() {
    }

    /**
     * The mean of each layer's next-token loss, weighted.
     *
     * @param model the decoder
     * @param tokens {@code [batch][steps]}
     * @param weights one weight per block (exit after block {@code i}); for example rising
     *     linearly
     * @return the weighted loss
     */
    public static Var deepSupervisionLoss(HookedDecoder model, int[][] tokens, float[] weights) {
        int batch = tokens.length;
        int n = tokens[0].length - 1;
        int[][] in = new int[batch][];
        int[][] target = new int[batch][];
        for (int b = 0; b < batch; b++) {
            in[b] = Arrays.copyOf(tokens[b], n);
            target[b] = Arrays.copyOfRange(tokens[b], 1, n + 1);
        }
        HookedDecoder.Run run = model.forward(in, null, null, true);
        Var total = null;
        float sum = 0;
        for (int i = 0; i < model.layers(); i++) {
            if (weights[i] == 0) {
                continue;
            }
            Var l = HookedDecoder.loss(model.project(run.hidden().get(i + 1)), target).mul(weights[i]);
            total = total == null ? l : total.add(l);
            sum += weights[i];
        }
        return total.div(sum);
    }

    /**
     * @param tokens the generated tokens
     * @param blocksRun the blocks run in total (out of {@code tokens * layers})
     * @param exits the exit layer (number of blocks run) of each token
     */
    public record Result(int[] tokens, int blocksRun, int[] exits) {

        /** @return the mean number of blocks per token */
        public double meanDepth() {
            return tokens.length == 0 ? 0 : (double) blocksRun / tokens.length;
        }
    }

    /**
     * CALM-style greedy decoding with a confidence exit.
     *
     * @param model the decoder
     * @param prompt the prompt
     * @param maxNew the tokens to generate
     * @param threshold the softmax confidence that ends a token's computation, for example 0.9
     * @return the tokens and the blocks run
     */
    public static Result generate(HookedDecoder model, int[] prompt, int maxNew, double threshold) {
        int[] context = prompt.clone();
        int[] exits = new int[maxNew];
        int total = 0;
        for (int n = 0; n < maxNew; n++) {
            HookedDecoder.Walk w = model.walk(new int[][] {context}, null, null, false);
            int token = -1;
            while (w.depth() < model.layers()) {
                Var h = w.advance();
                int len = context.length;
                float[] logits = model.project(h.slice(1, len - 1, len)).value().toArray();
                double[] p = LogitMath.softmax(logits);
                int best = LogitMath.argmax(logits);
                token = best;
                if (p[best] >= threshold) {
                    break;
                }
            }
            exits[n] = w.depth();
            total += w.depth();
            context = Arrays.copyOf(context, context.length + 1);
            context[context.length - 1] = token;
        }
        return new Result(Arrays.copyOfRange(context, prompt.length, context.length), total, exits);
    }

    /**
     * @param tokens the generated tokens
     * @param fullPasses the passes of the whole model (verification)
     * @param draftPasses the passes of the first blocks (drafting)
     */
    public record Speculation(int[] tokens, int fullPasses, int draftPasses) {
    }

    /**
     * Self-speculative greedy decoding: the first {@code exitLayer} blocks draft
     * {@code lookahead} tokens, the full model verifies them in one pass.
     *
     * @param model the decoder
     * @param prompt the prompt
     * @param maxNew the tokens to generate
     * @param exitLayer the blocks the draft runs
     * @param lookahead the tokens drafted per verification
     * @return the tokens, equal to full greedy decoding's
     */
    public static Speculation selfSpeculative(HookedDecoder model, int[] prompt, int maxNew, int exitLayer,
            int lookahead) {
        int[] context = prompt.clone();
        int vocab = model.vocabSize();
        int full = 0;
        int draft = 0;
        List<Integer> out = new ArrayList<>();
        while (out.size() < maxNew) {
            int[] guess = new int[lookahead];
            int[] d = context;
            for (int i = 0; i < lookahead; i++) {
                Var l = model.forward(new int[][] {d}, null, null, false, exitLayer).logits();
                draft++;
                int len = d.length;
                guess[i] = LogitMath.argmax(l.value().slice(1, len - 1, len).toArray());
                d = Arrays.copyOf(d, len + 1);
                d[len] = guess[i];
            }
            float[] logits = model.forward(new int[][] {d}).logits().value().toArray();
            full++;
            int base = context.length - 1;
            int accepted = 0;
            int next = LogitMath.argmax(Arrays.copyOfRange(logits, base * vocab, (base + 1) * vocab));
            while (accepted < lookahead && guess[accepted] == next) {
                accepted++;
                int pos = base + accepted;
                next = LogitMath.argmax(Arrays.copyOfRange(logits, pos * vocab, (pos + 1) * vocab));
            }
            int keep = context.length + accepted;
            for (int i = 0; i < accepted && out.size() < maxNew; i++) {
                out.add(guess[i]);
            }
            if (out.size() < maxNew) {
                out.add(next);
            }
            context = Arrays.copyOf(d, keep + 1);
            context[keep] = next;
        }
        return new Speculation(out.stream().mapToInt(Integer::intValue).toArray(), full, draft);
    }
}
