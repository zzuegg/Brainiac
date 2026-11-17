package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Token masking for masked image modeling (MAE, I-JEPA, BEiT): picks which patches each
 * sample keeps, gathers the kept (or the masked) tokens so an encoder only processes those,
 * and scatters a shorter sequence back into the full patch order. Gradients flow through the
 * gathers.
 */
public final class PatchMasking {

    /**
     * Which tokens each sample keeps and which it hides.
     *
     * @param keep per sample, the kept token indices
     * @param masked per sample, the hidden token indices
     */
    public record Mask(int[][] keep, int[][] masked) {
    }

    private PatchMasking() {
    }

    /**
     * A uniformly random mask per sample.
     *
     * @param batch the samples
     * @param tokens the tokens per sample
     * @param ratio the share of tokens hidden, such as 0.75 for MAE
     * @param random the random source
     * @return the mask
     */
    public static Mask random(int batch, int tokens, float ratio, RandomGenerator random) {
        int hidden = Math.max(1, Math.min(tokens - 1, Math.round(tokens * ratio)));
        int[][] keep = new int[batch][tokens - hidden];
        int[][] masked = new int[batch][hidden];
        for (int b = 0; b < batch; b++) {
            int[] perm = new int[tokens];
            for (int i = 0; i < tokens; i++) {
                perm[i] = i;
            }
            for (int i = tokens - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int t = perm[i];
                perm[i] = perm[j];
                perm[j] = t;
            }
            int[] k = java.util.Arrays.copyOfRange(perm, 0, tokens - hidden);
            int[] m = java.util.Arrays.copyOfRange(perm, tokens - hidden, tokens);
            java.util.Arrays.sort(k);
            java.util.Arrays.sort(m);
            keep[b] = k;
            masked[b] = m;
        }
        return new Mask(keep, masked);
    }

    /**
     * Picks tokens per sample.
     *
     * @param x {@code [batch, tokens, dim]}
     * @param indices per sample, the token indices to pick (the same count for every sample)
     * @return {@code [batch, count, dim]}
     */
    public static Var gather(Var x, int[][] indices) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        int d = x.shape().dim(2);
        int k = indices[0].length;
        float[] flat = new float[b * k];
        for (int s = 0; s < b; s++) {
            for (int j = 0; j < k; j++) {
                flat[s * k + j] = (float) s * t + indices[s][j];
            }
        }
        return x.reshape(b * t, d).gather(0, Tensor.of(Shape.of(b * k), flat, x.value().device())).reshape(b, k, d);
    }

    /**
     * Gathers the same token positions from a {@code [tokens, dim]} table for every sample,
     * such as position embeddings of the masked patches.
     *
     * @param table {@code [tokens, dim]}
     * @param indices per sample, the token indices
     * @return {@code [batch, count, dim]}
     */
    public static Var gatherRows(Var table, int[][] indices) {
        int b = indices.length;
        int k = indices[0].length;
        float[] flat = new float[b * k];
        for (int s = 0; s < b; s++) {
            for (int j = 0; j < k; j++) {
                flat[s * k + j] = indices[s][j];
            }
        }
        return table.gather(0, Tensor.of(Shape.of(b, k), flat, table.value().device()));
    }

    /**
     * Puts kept tokens and filler tokens back into the original order, as MAE's decoder
     * input: kept positions get {@code kept}, masked positions get {@code filler}.
     *
     * @param kept {@code [batch, keepCount, dim]}
     * @param filler {@code [batch, maskCount, dim]}, such as a broadcast mask token
     * @param mask the mask that split them
     * @return {@code [batch, tokens, dim]}
     */
    public static Var unshuffle(Var kept, Var filler, Mask mask) {
        int b = kept.shape().dim(0);
        int nk = kept.shape().dim(1);
        int nm = filler.shape().dim(1);
        int t = nk + nm;
        int[][] order = new int[b][t];
        for (int s = 0; s < b; s++) {
            for (int j = 0; j < nk; j++) {
                order[s][mask.keep()[s][j]] = j;
            }
            for (int j = 0; j < nm; j++) {
                order[s][mask.masked()[s][j]] = nk + j;
            }
        }
        return gather(Var.concat(1, kept, filler), order);
    }

    /**
     * A {@code [batch, tokens, 1]} tensor with 1 at masked tokens and 0 elsewhere, for losses
     * on masked tokens only.
     *
     * @param mask the mask
     * @param tokens the tokens per sample
     * @return the indicator
     */
    public static Tensor indicator(Mask mask, int tokens) {
        int b = mask.masked().length;
        float[] m = new float[b * tokens];
        for (int s = 0; s < b; s++) {
            for (int j : mask.masked()[s]) {
                m[s * tokens + j] = 1;
            }
        }
        return Tensor.of(Shape.of(b, tokens, 1), m);
    }
}
