package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.Arrays;

/**
 * Contrastive search (Su et al. 2022, "A Contrastive Framework for Neural Text Generation",
 * <a href="https://arxiv.org/abs/2202.06417">arXiv:2202.06417</a>): a deterministic decoder
 * that avoids the repetition loops of greedy search. Among the {@code k} most likely tokens
 * it picks the one maximizing {@code (1 - alpha) p(v) - alpha max_j cos(h_v, h_j)}: model
 * confidence minus a <em>degeneration penalty</em>, the largest cosine similarity between
 * the candidate's last hidden state and those of the tokens already in the text. With
 * {@code alpha = 0} it is greedy search.
 *
 * <p>The {@code k} candidates are scored in one batched forward pass of a
 * {@link HookedDecoder}, which exposes the hidden states.
 */
public final class ContrastiveSearch {

    private ContrastiveSearch() {
    }

    /**
     * Generates a continuation.
     *
     * @param model the decoder
     * @param prompt the prompt
     * @param maxNew the tokens to add
     * @param k the candidates per step, for example 4
     * @param alpha the degeneration penalty weight, for example 0.6
     * @return the new tokens
     */
    public static int[] generate(HookedDecoder model, int[] prompt, int maxNew, int k, double alpha) {
        int[] context = prompt.clone();
        int dim = model.settings().dim();
        for (int n = 0; n < maxNew; n++) {
            float[] logits = model.lastLogits(context);
            double[] p = LogitMath.softmax(logits);
            Integer[] order = new Integer[p.length];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Double.compare(p[b], p[a]));
            int kk = Math.min(k, p.length);
            int[][] batch = new int[kk][];
            for (int c = 0; c < kk; c++) {
                batch[c] = Arrays.copyOf(context, context.length + 1);
                batch[c][context.length] = order[c];
            }
            HookedDecoder.Run run = model.forward(batch);
            Var last = run.hidden().get(run.hidden().size() - 1);
            float[] h = last.value().toArray();
            int len = context.length + 1;
            int best = order[0];
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < kk; c++) {
                int candOff = (c * len + len - 1) * dim;
                double maxSim = -1;
                for (int j = 0; j < len - 1; j++) {
                    maxSim = Math.max(maxSim, cosine(h, candOff, (c * len + j) * dim, dim));
                }
                double score = (1 - alpha) * p[order[c]] - alpha * maxSim;
                if (score > bestScore) {
                    bestScore = score;
                    best = order[c];
                }
            }
            context = Arrays.copyOf(context, context.length + 1);
            context[context.length - 1] = best;
        }
        return Arrays.copyOfRange(context, prompt.length, context.length);
    }

    private static double cosine(float[] h, int a, int b, int dim) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < dim; i++) {
            dot += (double) h[a + i] * h[b + i];
            na += (double) h[a + i] * h[a + i];
            nb += (double) h[b + i] * h[b + i];
        }
        return dot / Math.sqrt(na * nb + 1e-12);
    }
}
