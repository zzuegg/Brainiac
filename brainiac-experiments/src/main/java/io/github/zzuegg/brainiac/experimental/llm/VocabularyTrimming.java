package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.TreeSet;

/**
 * Vocabulary trimming (Ushio et al. 2023, "An Efficient Multilingual Language Model
 * Compression through Vocabulary Trimming", <a href="https://arxiv.org/abs/2305.15020">arXiv:2305.15020</a>):
 * a multilingual or general model spends most of its embedding and output matrices on
 * tokens a target domain never uses. Keeping only the tokens a corpus contains (plus
 * specials) shrinks both matrices and the softmax, and the trimmed model's scores of the
 * kept tokens equal the original's, so it is lossless on that domain up to the softmax
 * renormalization over fewer tokens.
 */
public final class VocabularyTrimming {

    private VocabularyTrimming() {
    }

    /**
     * @param model the smaller decoder
     * @param newToOld the original id of each new id
     * @param oldToNew the new id of each original id, -1 if trimmed
     */
    public record Trimmed(HookedDecoder model, int[] newToOld, int[] oldToNew) {

        /**
         * @param tokens original ids, all kept
         * @return the new ids
         */
        public int[] encode(int[] tokens) {
            int[] out = new int[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                out[i] = oldToNew[tokens[i]];
                if (out[i] < 0) {
                    throw new IllegalArgumentException("token " + tokens[i] + " was trimmed");
                }
            }
            return out;
        }

        /**
         * @param tokens new ids
         * @return the original ids
         */
        public int[] decode(int[] tokens) {
            int[] out = new int[tokens.length];
            for (int i = 0; i < tokens.length; i++) {
                out[i] = newToOld[tokens[i]];
            }
            return out;
        }
    }

    /**
     * @param corpus token sequences of the target domain
     * @param always tokens to keep regardless (end of sequence, padding)
     * @return the sorted ids to keep
     */
    public static int[] used(int[][] corpus, int... always) {
        TreeSet<Integer> keep = new TreeSet<>();
        for (int[] row : corpus) {
            for (int t : row) {
                keep.add(t);
            }
        }
        for (int t : always) {
            keep.add(t);
        }
        return keep.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * A copy of {@code model} over the kept tokens only. The projections and head must be
     * the original {@code Dense} layers.
     *
     * @param model the decoder
     * @param keep the original ids to keep, ascending
     * @return the trimmed decoder and the id maps
     */
    public static Trimmed trim(HookedDecoder model, int[] keep) {
        HookedDecoder.Settings s = model.settings();
        HookedDecoder small = HookedDecoder.create(new HookedDecoder.Settings(keep.length, s.dim(), s.layers(),
                s.heads(), s.ffnMultiplier(), s.ropeBase()), 0);
        List<Var> from = model.baseParameters();
        List<Var> to = small.baseParameters();
        Tensor idx = ids(keep);
        // the embedding is first, the head's weight and bias last
        to.get(0).assign(from.get(0).value().gather(0, idx));
        for (int i = 1; i < from.size() - 2; i++) {
            to.get(i).assign(from.get(i).value().add(0f));
        }
        if (!(model.head() instanceof Dense)) {
            throw new IllegalArgumentException("the head must be a Dense layer");
        }
        Tensor w = from.get(from.size() - 2).value();
        to.get(to.size() - 2).assign(w.gather(1, idx));
        to.get(to.size() - 1).assign(from.get(from.size() - 1).value().gather(0, idx));
        int[] oldToNew = new int[model.vocabSize()];
        Arrays.fill(oldToNew, -1);
        for (int i = 0; i < keep.length; i++) {
            oldToNew[keep[i]] = i;
        }
        return new Trimmed(small, keep.clone(), oldToNew);
    }

    private static Tensor ids(int[] keep) {
        float[] f = new float[keep.length];
        for (int i = 0; i < keep.length; i++) {
            f[i] = keep[i];
        }
        return Tensor.of(Shape.of(keep.length), f);
    }
}
