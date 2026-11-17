package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.InputRecorder;
import io.github.zzuegg.brainiac.experimental.nn.LowRankDense;
import io.github.zzuegg.brainiac.experimental.nn.WeightPruning;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * One-shot compression of every projection of a trained decoder, from one calibration
 * batch, compared by the language-model loss afterwards: magnitude, Wanda and SparseGPT
 * pruning (unstructured and 2:4), and plain versus activation-aware SVD factorization.
 */
public final class CompressionComparison {

    /** The compression methods. */
    public enum Method {
        /** smallest weights per unit */
        MAGNITUDE,
        /** weight times input norm */
        WANDA,
        /** optimal brain surgeon with weight updates */
        SPARSEGPT,
        /** 2:4 by Wanda's score */
        WANDA_2_4,
        /** 2:4 with SparseGPT's updates */
        SPARSEGPT_2_4,
        /** truncated SVD */
        SVD,
        /** activation-aware SVD */
        ACTIVATION_AWARE_SVD
    }

    private CompressionComparison() {
    }

    /**
     * Compresses a copy of {@code base}.
     *
     * @param base the trained decoder (not changed)
     * @param method the method
     * @param amount the sparsity for unstructured pruning, or the share of the rank kept for SVD
     * @param seed the calibration seed
     * @return the compressed copy
     */
    public static HookedDecoder compress(HookedDecoder base, Method method, double amount, long seed) {
        HookedDecoder m = base.copy();
        Map<String, InputRecorder> rec = new HashMap<>();
        m.replaceLinears((name, l) -> {
            InputRecorder r = new InputRecorder(l);
            rec.put(name, r);
            return r;
        });
        m.forward(CountingTasks.sequences(new SplittableRandom(seed), 16, 12, m.vocabSize(), 1, 2));
        m.replaceLinears((name, l) -> {
            Dense d = (Dense) ((InputRecorder) l).base();
            Tensor x = rec.get(name).inputs();
            Tensor w = d.weight().value();
            int rank = Math.max(1, (int) Math.round(amount * Math.min(w.shape().dim(0), w.shape().dim(1))));
            Layer out = switch (method) {
                case MAGNITUDE -> WeightPruning.replaceWeight(d, WeightPruning.magnitude(w, amount));
                case WANDA -> WeightPruning.replaceWeight(d, WeightPruning.wanda(w, x, amount));
                case SPARSEGPT -> WeightPruning.replaceWeight(d, WeightPruning.sparseGpt(w, x, amount, 0, 0, 32, 0.01));
                case WANDA_2_4 -> WeightPruning.replaceWeight(d, WeightPruning.semiStructured(w, x, 2, 4));
                case SPARSEGPT_2_4 -> WeightPruning.replaceWeight(d, WeightPruning.sparseGpt(w, x, 0, 2, 4, 32, 0.01));
                case SVD -> LowRankDense.factorize(d, rank);
                case ACTIVATION_AWARE_SVD -> LowRankDense.activationAware(d, x, rank);
            };
            return out;
        });
        return m;
    }
}
