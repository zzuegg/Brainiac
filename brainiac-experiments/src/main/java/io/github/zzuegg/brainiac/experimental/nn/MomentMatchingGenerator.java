package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MeanFlow;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Inductive moment matching (IMM, Zhou, Ermon and Song,
 * <a href="https://arxiv.org/abs/2503.07565">arXiv:2503.07565</a>): a single-stage few-step
 * generator {@code f(x_t, s, t)} that jumps from time {@code t} to any earlier time {@code s}
 * along core {@code FlowMatching}'s straight interpolant ({@code x_t = (1 - t) x0 + t e}). It is
 * trained without a teacher and without ODE simulation: for a group of samples sharing
 * {@code (s, r, t)} with {@code s < r < t}, the distribution of {@code f(x_t, s, t)} must match
 * that of {@code f(x_r, s, r)} (the stop-gradient target, one step closer to {@code s}), and
 * the match is scored by the maximum mean discrepancy (MMD) between the two particle groups
 * with a Laplace kernel. Matching distributions instead of individual trajectories (as
 * consistency models do) is what keeps training stable across seeds.
 *
 * <p>The jump uses the "Euler" parameterisation {@code f = x_t - (t - s) u(x_t, s, t)} of a
 * {@link MeanFlow.Network}, so the same networks serve MeanFlow and IMM.
 */
public final class MomentMatchingGenerator {
    private final MeanFlow.Network net;
    private final int group;
    private final float gap;
    private final float bandwidth;

    private MomentMatchingGenerator(MeanFlow.Network net, int group, float gap, float bandwidth) {
        this.net = net;
        this.group = group;
        this.gap = gap;
        this.bandwidth = bandwidth;
    }

    /**
     * Creates the trainer.
     *
     * @param net the jump network {@code u(x, s, t)}
     * @param group the particles per MMD group, such as 4 or 8
     * @param gap the step {@code t - r}, such as 1/16
     * @param bandwidth the Laplace kernel bandwidth, in data units
     * @return the generator
     */
    public static MomentMatchingGenerator of(MeanFlow.Network net, int group, float gap, float bandwidth) {
        if (group < 2) {
            throw new IllegalArgumentException("MMD needs at least two particles per group");
        }
        return new MomentMatchingGenerator(net, group, gap, bandwidth);
    }

    /** @return the jump network */
    public MeanFlow.Network network() {
        return net;
    }

    private Var jump(Var x, Tensor s, Tensor t) {
        return x.sub(net.apply(x, Var.of(s), Var.of(t)).mul(t.sub(s)));
    }

    /**
     * The IMM loss on a batch whose size is a multiple of the group size.
     *
     * @param x0 data {@code [batch, dim]}
     * @param random the source of times and noise
     * @return the mean MMD^2 over groups
     */
    public Var loss(Tensor x0, RandomGenerator random) {
        int b = x0.shape().dim(0);
        int dim = x0.shape().dim(1);
        int groups = b / group;
        float[] t = new float[b];
        float[] r = new float[b];
        float[] s = new float[b];
        for (int g = 0; g < groups; g++) {
            float tg = (float) (0.05 + 0.95 * random.nextDouble());
            float sg = (float) (random.nextDouble() * tg);
            float rg = Math.max(sg, tg - gap);
            for (int k = 0; k < group; k++) {
                t[g * group + k] = tg;
                r[g * group + k] = rg;
                s[g * group + k] = sg;
            }
        }
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random);
        Tensor tc = GenerativeOps.column(t);
        Tensor rc = GenerativeOps.column(r);
        Tensor sc = GenerativeOps.column(s);
        Tensor xt = x0.mul(tc.neg().add(1f)).add(noise.mul(tc));
        Tensor xr = x0.mul(rc.neg().add(1f)).add(noise.mul(rc));
        Var yt = jump(Var.of(xt), sc, tc);
        Tensor yr = jump(Var.of(xr), sc, rc).value();
        Var total = null;
        for (int g = 0; g < groups; g++) {
            Var a = yt.slice(0, g * group, (g + 1) * group);
            Var c = Var.of(yr.slice(0, g * group, (g + 1) * group));
            Var mmd = kernel(a, a, dim).mean().add(kernel(c, c, dim).mean()).sub(kernel(a, c, dim).mean().mul(2f));
            total = total == null ? mmd : total.add(mmd);
        }
        return total.div(groups);
    }

    /** The Laplace kernel matrix {@code exp(-|a_i - b_j| / (bandwidth * dim))}, {@code [n, m]}. */
    private Var kernel(Var a, Var b, int dim) {
        int n = a.shape().dim(0);
        int m = b.shape().dim(0);
        Var diff = a.reshape(n, 1, dim).sub(b.reshape(1, m, dim));
        Var dist = diff.square().sum(2, false).add(1e-8f).sqrt();
        return dist.div(-bandwidth * dim).exp();
    }

    /**
     * Pushforward sampling: jumps from {@code t = 1} through the given times to 0.
     *
     * @param noise standard normal noise {@code [n, dim]}
     * @param steps the number of jumps, 1 for one-step generation
     * @return the samples
     */
    public Tensor sample(Tensor noise, int steps) {
        int n = noise.shape().dim(0);
        Tensor x = noise;
        for (int i = 0; i < steps; i++) {
            float t = 1 - (float) i / steps;
            float s = 1 - (float) (i + 1) / steps;
            x = jump(Var.of(x), Tensor.full(Shape.of(n, 1), s), Tensor.full(Shape.of(n, 1), t)).value();
        }
        return x;
    }
}
