package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * The gated delta rule with a <b>per-channel</b> decay, shared by Kimi Delta Attention,
 * DeltaProduct and their tests:
 *
 * <pre>
 * S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T,    o_t = S_t^T q_t
 * </pre>
 *
 * <p>with {@code S} of shape {@code [dk, dv]} per head. A scalar decay is the special case of
 * equal channels.
 */
final class DeltaRuleOps {

    private DeltaRuleOps() {
    }

    /**
     * Token by token.
     *
     * @param q {@code [b, h, t, dk]}
     * @param k {@code [b, h, t, dk]}
     * @param v {@code [b, h, t, dv]}
     * @param logAlpha per-channel log decay {@code [b, h, t, dk]} (at most 0)
     * @param beta {@code [b, h, t]}
     * @return {@code [b, h, t, dv]}
     */
    static Var recurrent(Var q, Var k, Var v, Var logAlpha, Var beta) {
        int b = q.shape().dim(0);
        int h = q.shape().dim(1);
        int t = q.shape().dim(2);
        int dk = k.shape().dim(3);
        int dv = v.shape().dim(3);
        Var state = Var.of(Tensor.zeros(Shape.of(b, h, dk, dv)));
        List<Var> out = new ArrayList<>();
        for (int s = 0; s < t; s++) {
            Var ks = k.slice(2, s, s + 1);
            Var be = beta.slice(2, s, s + 1).reshape(b, h, 1, 1);
            state = state.mul(logAlpha.slice(2, s, s + 1).exp().transpose());
            Var predicted = ks.matmul(state);
            state = state.add(ks.transpose().matmul(v.slice(2, s, s + 1).sub(predicted)).mul(be));
            out.add(q.slice(2, s, s + 1).matmul(state));
        }
        return Var.concat(2, out.toArray(Var[]::new));
    }

    /**
     * Chunkwise parallel: inside a chunk of {@code c} steps, with cumulative log decays
     * {@code G}, the pairwise decays {@code D[t, i, :] = exp(G_t - G_i)} (stable, exponents at
     * most 0) give {@code A_ti = sum_d k_t k_i D[t, i]}, the pseudo-values solve
     * {@code (I + diag(beta) A) U = diag(beta)(V - (K o e^G) S_0)}, and
     * {@code O = (Q o e^G) S_0 + (sum_d q_t k_i D[t, i]) U}. The chunk's end state carries
     * over.
     *
     * @param q {@code [b, h, t, dk]}
     * @param k {@code [b, h, t, dk]}
     * @param v {@code [b, h, t, dv]}
     * @param logAlpha per-channel log decay {@code [b, h, t, dk]}
     * @param beta {@code [b, h, t]}
     * @param chunk the chunk length
     * @return {@code [b, h, t, dv]}
     */
    static Var chunkwise(Var q, Var k, Var v, Var logAlpha, Var beta, int chunk) {
        int b = q.shape().dim(0);
        int h = q.shape().dim(1);
        int t = q.shape().dim(2);
        int dk = k.shape().dim(3);
        int dv = v.shape().dim(3);
        Var state = Var.of(Tensor.zeros(Shape.of(b, h, dk, dv)));
        List<Var> out = new ArrayList<>();
        for (int start = 0; start < t; start += chunk) {
            int end = Math.min(t, start + chunk);
            int c = end - start;
            Var qc = q.slice(2, start, end);
            Var kc = k.slice(2, start, end);
            Var vc = v.slice(2, start, end);
            Var bc = beta.slice(2, start, end).reshape(b, h, c, 1);
            Var g = logAlpha.slice(2, start, end).cumsum(2);                          // [b, h, c, dk]
            Tensor lower = MixerOps.lowerOnes(c).reshape(c, c, 1);
            Var diff = g.reshape(b, h, c, 1, dk).sub(g.reshape(b, h, 1, c, dk)).mul(lower);
            Var decay = diff.exp().mul(lower);                                      // [b, h, c, c, dk]
            Var kk = kc.reshape(b, h, c, 1, dk).mul(kc.reshape(b, h, 1, c, dk)).mul(decay).sum(-1, false);
            Var qk = qc.reshape(b, h, c, 1, dk).mul(kc.reshape(b, h, 1, c, dk)).mul(decay).sum(-1, false);
            Tensor strict = MixerOps.lowerOnes(c).sub(identity(c));
            Var inverse = GatedDeltaNet.unitLowerInverse(kk.mul(strict).mul(bc), c);
            Var eg = g.exp();
            Var u = inverse.matmul(vc.sub(kc.mul(eg).matmul(state)).mul(bc));            // [b, h, c, dv]
            out.add(qc.mul(eg).matmul(state).add(qk.matmul(u)));
            Var last = g.slice(2, c - 1, c);                                         // [b, h, 1, dk]
            Var toEnd = last.sub(g).exp();                                           // [b, h, c, dk]
            state = state.mul(last.exp().transpose()).add(kc.mul(toEnd).transpose().matmul(u));
        }
        return out.size() == 1 ? out.getFirst() : Var.concat(2, out.toArray(Var[]::new));
    }

    static Tensor identity(int c) {
        float[] m = new float[c * c];
        for (int i = 0; i < c; i++) {
            m[i * c + i] = 1;
        }
        return Tensor.of(Shape.of(c, c), m);
    }

    /** A short causal convolution kernel {@code [4, features]} that starts close to the identity. */
    static Var convInit(int features, java.util.random.RandomGenerator random) {
        float[] w = new float[4 * features];
        for (int i = 0; i < w.length; i++) {
            w[i] = (i < features ? 1f : 0f) + (float) (random.nextDouble() * 0.2 - 0.1);
        }
        return Var.param(Tensor.of(Shape.of(4, features), w));
    }

    /** Linear map, short causal convolution, SiLU. */
    static Var convSilu(Var x, Var w, Var conv) {
        return MixerOps.causalDepthwise(x.matmul(w), conv)
                .activation(io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SILU, 0);
    }
}
