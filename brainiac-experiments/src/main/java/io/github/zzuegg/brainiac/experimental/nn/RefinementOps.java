package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Shared helpers of the attention, residual and MoE refinements: 0/1 keep masks, the
 * pre-norm residual stack plumbing and plain softmax attention over given keys and values.
 */
final class RefinementOps {

    private RefinementOps() {
    }

    /**
     * A {@code [queries, prefix + keys]} 0/1 mask. The first {@code prefix} columns (registers,
     * cached steps) are always visible; the others obey causality, offset by {@code offset}
     * (the global position of query 0 relative to key 0), and a sliding window.
     *
     * @param queries the queries
     * @param keys the keys after the prefix
     * @param prefix leading always-visible keys
     * @param offset query {@code i} sits at key position {@code i + offset}
     * @param causal whether later keys are hidden
     * @param window how many most recent keys a query sees (itself included), or 0 for all
     * @param device the device
     * @return the mask
     */
    static Tensor keep(int queries, int keys, int prefix, int offset, boolean causal, int window, Device device) {
        int cols = prefix + keys;
        float[] m = new float[queries * cols];
        for (int i = 0; i < queries; i++) {
            int pos = i + offset;
            for (int j = 0; j < prefix; j++) {
                m[i * cols + j] = 1;
            }
            for (int j = 0; j < keys; j++) {
                boolean visible = (!causal || j <= pos) && (window <= 0 || j > pos - window);
                m[i * cols + prefix + j] = visible ? 1 : 0;
            }
        }
        return Tensor.of(Shape.of(queries, cols), m, device);
    }

    /** {@code 0} where {@code keep} is 1, very negative where it is 0: an additive softmax mask. */
    static Tensor additive(Tensor keep) {
        return keep.sub(1f).mul(1e9f);
    }

    /** RMS norm over the last axis with a learned gain. */
    static Var rms(Var x, Var gain) {
        return MixerOps.rmsNorm(x).mul(gain);
    }

    /** A gain vector of ones. */
    static Var ones(int features) {
        return MixerOps.filled(1f, features);
    }

    /**
     * Causal softmax attention over split heads, with optional RoPE on queries and keys.
     *
     * @param q {@code [b, h, t, d]}
     * @param k {@code [b, h, t, d]}
     * @param v {@code [b, h, t, d]}
     * @param rotaryBase the RoPE base, or 0 for none
     * @param window the sliding window, or 0 for full causal attention
     * @return {@code [b, h, t, d]}
     */
    static Var causalAttention(Var q, Var k, Var v, float rotaryBase, int window) {
        if (rotaryBase > 0) {
            q = Functional.rotary(q, rotaryBase, 0);
            k = Functional.rotary(k, rotaryBase, 0);
        }
        int t = q.shape().dim(2);
        Tensor mask = additive(keep(t, t, 0, 0, true, window, q.value().device()));
        return TechniqueOps.attend(q, k, v, null, mask, (float) (1 / Math.sqrt(q.shape().dim(3))));
    }

    /** Number of whole elements of a tensor of this shape, as an int. */
    static int size(Shape s) {
        return Math.toIntExact(s.size());
    }
}
