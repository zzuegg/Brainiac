package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A Mamba-2 block computed with the state space duality (SSD) algorithm, plus the
 * Mamba-3-style complex (rotary) state variant.
 *
 * <p><b>Mamba-2.</b> Every head {@code h} has a scalar decay {@code A_h < 0}, so the
 * recurrence is {@code H_t = exp(delta_t A_h) H_{t-1} + delta_t B_t x_t^T} with a matrix state
 * {@code [N, P]} and output {@code y_t = H_t^T C_t}. Such a scalar-times-identity transition
 * makes the SSM a masked form of linear attention, {@code Y = (C B^T . L) (delta X)} with the
 * decay mask {@code L_ts = exp(sum_{s<j<=t} delta_j A)}. {@link #forward(Var)} uses the
 * chunked SSD algorithm: inside each chunk the quadratic attention-like form (matmuls),
 * between chunks the linear recurrence over chunk states. {@link #recurrent(Var)} runs the
 * plain recurrence; both agree.
 *
 * <p><b>Mamba-3 (rotary).</b> With {@link Variant#ROTARY} the state transition becomes
 * {@code exp(delta A) R(theta_t)}: a decay times a data-dependent rotation of each pair of
 * state channels, i.e. a complex-valued diagonal SSM. Real non-negative decays cannot count
 * modulo 2; rotations can (rotate by pi on every 1 bit), so this variant solves state-tracking
 * tasks like parity that Mamba-2 fails. It is computed exactly like RoPE: with cumulative
 * angles {@code phi_t}, rotating {@code B_t} and {@code C_t} by {@code -phi_t} turns the
 * rotating recurrence into a real SSD over the rotated inputs, so the chunked algorithm
 * still applies. {@link #recurrent(Var)} instead rotates the state step by step.
 *
 * <p>The block: input projection to {@code x} and gate {@code z}, a short causal convolution
 * and SiLU on {@code x}, the SSM with skip {@code D x}, RMS norm, {@code * silu(z)}, output
 * projection.
 *
 * <p>Papers: Dao and Gu, "Transformers are SSMs" (Mamba-2),
 * <a href="https://arxiv.org/abs/2405.21060">arXiv:2405.21060</a>; "Mamba-3",
 * <a href="https://arxiv.org/abs/2603.15569">arXiv:2603.15569</a>.
 */
public final class Mamba2Layer implements Layer {

    /** The state transition. */
    public enum Variant {
        /** Real scalar decay per head (Mamba-2). */
        REAL,
        /** Decay times a data-dependent rotation of state pairs (Mamba-3 complex state). */
        ROTARY
    }

    private final int features;
    private final int heads;
    private final int headSize;
    private final int state;
    private final int chunk;
    private final Variant variant;
    private final Var inProj;
    private final Var conv;
    private final Var wB;
    private final Var wC;
    private final Var wDelta;
    private final Var deltaBias;
    private final Var aLog;
    private final Var wTheta;
    private final Var skip;
    private final Var outProj;

    /**
     * Creates the block.
     *
     * @param features the model width
     * @param heads the number of heads
     * @param headSize the width {@code P} of each head; the inner width is {@code heads * headSize}
     * @param state the state size {@code N}, even for the rotary variant
     * @param chunk the SSD chunk length
     * @param variant real (Mamba-2) or rotary (Mamba-3) states
     * @param random the random source
     */
    public Mamba2Layer(int features, int heads, int headSize, int state, int chunk, Variant variant,
            RandomGenerator random) {
        if (variant == Variant.ROTARY && state % 2 != 0) {
            throw new IllegalArgumentException("the rotary variant needs an even state size");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = headSize;
        this.state = state;
        this.chunk = chunk;
        this.variant = variant;
        int inner = heads * headSize;
        this.inProj = MixerOps.weight(random, features, 2 * inner);
        this.conv = MixerOps.uniform(random, 0.5f, 4, inner);
        this.wB = MixerOps.weight(random, features, state);
        this.wC = MixerOps.weight(random, features, state);
        this.wDelta = MixerOps.uniform(random, 0.1f, features, heads);
        this.deltaBias = MixerOps.filled(-1f, heads);
        float[] a = new float[heads];
        for (int h = 0; h < heads; h++) {
            // rotary states start with slow decay, so rotations can carry information far
            a[h] = (float) Math.log(variant == Variant.ROTARY ? 0.05 * (1 + h) : 1 + h);
        }
        this.aLog = Var.param(Tensor.of(Shape.of(heads), a));
        this.wTheta = MixerOps.uniform(random, 1f, features, heads * state / 2);
        this.skip = MixerOps.filled(1f, heads, 1, 1);
        this.outProj = MixerOps.weight(random, inner, features);
    }

    /**
     * Describes the block for {@code Model.sequential().add(...)}.
     *
     * @param heads the number of heads
     * @param headSize the width of each head
     * @param state the state size
     * @param chunk the SSD chunk length
     * @param variant real or rotary states
     * @return the spec
     */
    public static LayerSpec of(int heads, int headSize, int state, int chunk, Variant variant) {
        return MixerOps.sameShape((in, r) -> new Mamba2Layer(in.dim(-1), heads, headSize, state, chunk, variant, r));
    }

    /** Projections: x [b,H,t,P], z [b,t,inner], logA [b,H,t], delta [b,H,t,1], B, C [b,H|1,t,N], theta [b,H,t,N/2]. */
    private Var[] project(Var input) {
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int inner = heads * headSize;
        Var xz = input.matmul(inProj);
        Var x = MixerOps.causalDepthwise(xz.slice(2, 0, inner), conv).activation(ActivationOp.SILU, 0);
        Var z = xz.slice(2, inner, 2 * inner);
        Var delta = input.matmul(wDelta).add(deltaBias).softplus().permute(0, 2, 1);   // [b, H, t]
        Var logA = delta.mul(aLog.exp().neg().reshape(heads, 1));
        Var bb = input.matmul(wB).reshape(b, 1, t, state);
        Var cc = input.matmul(wC).reshape(b, 1, t, state);
        Var theta = variant == Variant.ROTARY ? MixerOps.split(input.matmul(wTheta), heads) : null;
        return new Var[] {MixerOps.split(x, heads), z, logA, delta.reshape(b, heads, t, 1), bb, cc, theta};
    }

    /** Rotates pairs (i, i + N/2) of the last axis by angles {@code [..., N/2]}. */
    private static Var rotate(Var v, Var cos, Var sin) {
        int n = v.shape().dim(-1);
        Var v1 = v.slice(-1, 0, n / 2);
        Var v2 = v.slice(-1, n / 2, n);
        return Var.concat(-1, v1.mul(cos).sub(v2.mul(sin)), v1.mul(sin).add(v2.mul(cos)));
    }

    private Var finish(Var[] p, Var y) {
        Var out = y.add(p[0].mul(skip));   // [b, H, t, P]
        return MixerOps.rmsNorm(MixerOps.merge(out)).mul(p[1].activation(ActivationOp.SILU, 0)).matmul(outProj);
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "Mamba2Layer");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var[] p = project(input);
        Var bb = p[4];
        Var cc = p[5];
        if (variant == Variant.ROTARY) {
            // cumulative angles; rotate B and C by -phi_t
            Var phi = p[6].permute(0, 1, 3, 2).matmul(Var.of(MixerOps.cumsumMatrix(t))).permute(0, 1, 3, 2);
            Var cos = MixerOps.cos(phi);
            Var sin = MixerOps.sin(phi).neg();
            bb = rotate(bb, cos, sin);   // [b, H, t, N]
            cc = rotate(cc, cos, sin);
        }
        Var dx = p[0].mul(p[3]);         // delta_t x_t, [b, H, t, P]
        Var h = Var.of(Tensor.zeros(Shape.of(b, heads, state, headSize)));
        List<Var> ys = new ArrayList<>();
        for (int start = 0; start < t; start += chunk) {
            int end = Math.min(t, start + chunk);
            int c = end - start;
            Var la = p[2].slice(2, start, end).matmul(Var.of(MixerOps.cumsumMatrix(c)));   // [b, H, c]
            Var bc = bb.slice(2, start, end);
            Var ccC = cc.slice(2, start, end);
            Var dxc = dx.slice(2, start, end);
            Var decay = MixerOps.segmentDecay(la);                                       // [b, H, c, c]
            Var intra = ccC.matmul(bc.transpose()).mul(decay).matmul(dxc);
            Var inter = ccC.matmul(h).mul(la.exp().reshape(b, heads, c, 1));
            ys.add(intra.add(inter));
            Var last = la.slice(2, c - 1, c);                                            // [b, H, 1]
            Var toEnd = last.sub(la).exp().reshape(b, heads, c, 1);
            h = h.mul(last.exp().reshape(b, heads, 1, 1)).add(bc.mul(toEnd).transpose().matmul(dxc));
        }
        return finish(p, ys.size() == 1 ? ys.getFirst() : Var.concat(2, ys.toArray(Var[]::new)));
    }

    /**
     * Computes the output with the plain recurrence (rotating the state for the rotary variant).
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}
     */
    public Var recurrent(Var input) {
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var[] p = project(input);
        Var h = Var.of(Tensor.zeros(Shape.of(b, heads, state, headSize)));
        List<Var> ys = new ArrayList<>();
        for (int s = 0; s < t; s++) {
            Var a = p[2].slice(2, s, s + 1).exp().reshape(b, heads, 1, 1);
            if (variant == Variant.ROTARY) {
                // rotate the state's N axis by theta_t: transpose so N is last
                Var th = p[6].slice(2, s, s + 1);   // [b, H, 1, N/2]
                h = rotate(h.transpose(), MixerOps.cos(th), MixerOps.sin(th)).transpose();
            }
            Var bs = p[4].slice(2, s, s + 1);       // [b, 1, 1, N]
            Var cs = p[5].slice(2, s, s + 1);
            Var dx = p[0].slice(2, s, s + 1).mul(p[3].slice(2, s, s + 1));   // [b, H, 1, P]
            h = h.mul(a).add(bs.transpose().matmul(dx));
            ys.add(cs.matmul(h));                   // [b, H, 1, P]
        }
        return finish(p, Var.concat(2, ys.toArray(Var[]::new)));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(inProj, conv, wB, wC, wDelta, deltaBias, aLog, skip, outProj));
        if (variant == Variant.ROTARY) {
            p.add(wTheta);
        }
        return p;
    }

    @Override
    public String toString() {
        return "Mamba2Layer(" + features + ", heads=" + heads + ", state=" + state + ", " + variant + ")";
    }
}
