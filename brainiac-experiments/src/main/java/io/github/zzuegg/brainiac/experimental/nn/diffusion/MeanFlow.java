package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.nn.diffusion.FlowMatching;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * One- and few-step generators that learn the <em>average</em> velocity of a flow over an
 * interval instead of the instantaneous one, so a single large step lands where many small
 * steps would. Both use the conventions of core {@link FlowMatching}: time 0 is data, time 1
 * is noise, {@code z_t = (1 - t) x + t e}, instantaneous velocity {@code v = e - x}. A
 * {@link Network} predicts {@code u(z_t, r, t)}, the average velocity from time {@code t}
 * back to {@code r <= t}, and a step is {@code z_r = z_t - (t - r) u(z_t, r, t)}.
 *
 * <ul>
 *   <li>{@link #meanFlowLoss}: MeanFlow (Geng et al., <a href="https://arxiv.org/abs/2505.13447">2505.13447</a>).
 *       The average velocity satisfies the identity {@code u = v - (t - r) du/dt}, with the
 *       total derivative {@code du/dt = v . d_z u + d_t u}, a Jacobian-vector product. The
 *       target {@code v - (t - r) jvp} is held fixed (stop-gradient). The product is exact:
 *       it comes from two reverse passes with core {@link Var#gradients(Var, Var, List, boolean)}
 *       (the gradient of a vector-Jacobian product with respect to its seed), not from finite
 *       differences. Losses are adaptively weighted by {@code 1 / (|err|^2 + c)^p}.</li>
 *   <li>{@link #shortcutLoss}: shortcut models (Frans et al., <a href="https://arxiv.org/abs/2410.12557">2410.12557</a>).
 *       Plain flow matching at step size 0 ({@code r = t}), plus self-consistency: one step of
 *       size {@code 2d} must equal two chained steps of size {@code d} (stop-gradient target).
 *       Needs no Jacobian, only forward passes.</li>
 * </ul>
 *
 * <p>With {@code r = t} either network is an ordinary velocity field, so
 * {@link #instantaneous} hands it to the core multi-step samplers.
 */
public final class MeanFlow {
    private static final int TIME_FEATURES = 16;
    private static final double MAX_FREQUENCY = 8;

    private MeanFlow() {
    }

    /** A network for the average velocity over an interval. */
    public interface Network {

        /**
         * Predicts the average velocity.
         *
         * @param x the points at time {@code t}, {@code [batch, dimension]}
         * @param r the interval starts, {@code [batch, 1]}
         * @param t the interval ends, {@code [batch, 1]}, at least {@code r}
         * @return {@code [batch, dimension]}
         */
        Var apply(Var x, Var r, Var t);

        /** @return the trainable parameters */
        List<Var> parameters();

        /** @return the sample size */
        int dimension();
    }

    /**
     * A residual MLP over the sample and sinusoidal features of {@code t} and {@code t - r},
     * built only from operations that have second derivatives, so {@link #meanFlowLoss} can
     * take its Jacobian-vector product.
     *
     * @param dimension the sample size
     * @param width the hidden width
     * @param depth the number of residual hidden layers
     * @param seed the initialization seed
     * @return the network
     */
    public static Network mlp(int dimension, int width, int depth, long seed) {
        return new IntervalMlp(dimension, width, depth, seed);
    }

    /**
     * The MeanFlow loss on a batch.
     *
     * @param net the average-velocity network
     * @param x0 data, {@code [batch, dimension]}
     * @param equalShare the share of pairs with {@code r = t} (plain flow matching), for example 0.5
     * @param adaptivePower the power {@code p} of the adaptive weight; 0 for a plain squared error
     * @param random the source of times and noise
     * @return the loss, a scalar
     */
    public static Var meanFlowLoss(Network net, Tensor x0, float equalShare, float adaptivePower,
            RandomGenerator random) {
        int batch = x0.shape().dim(0);
        int dim = x0.shape().dim(1);
        float[] r = new float[batch];
        float[] t = new float[batch];
        for (int i = 0; i < batch; i++) {
            float a = (float) random.nextDouble();
            float b = (float) random.nextDouble();
            t[i] = Math.max(a, b);
            r[i] = random.nextDouble() < equalShare ? t[i] : Math.min(a, b);
        }
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random);
        Tensor tc = column(t);
        Tensor zt = x0.mul(tc.neg().add(1f)).add(noise.mul(tc));
        Tensor v = noise.sub(x0);

        Var z = Var.param(zt);
        Var tv = Var.param(tc);
        Var u = net.apply(z, Var.of(column(r)), tv);
        // exact JVP along (v, dt = 1): differentiate the VJP J^T w with respect to its seed w
        Var w = Var.param(Tensor.zeros(Shape.of(batch, dim)));
        List<Var> vjp = Var.gradients(u, w, List.of(z, tv), true);
        Var directional = vjp.get(0).mul(Var.of(v)).sum().add(vjp.get(1).sum());
        Tensor jvp = Var.gradients(directional, List.of(w), false).get(0).value();

        float[] gap = new float[batch];
        for (int i = 0; i < batch; i++) {
            gap[i] = t[i] - r[i];
        }
        Tensor target = v.sub(jvp.mul(column(gap)));
        Var err = u.sub(Var.of(target)).square().sum(1, true);   // [batch, 1]
        return weighted(err, adaptivePower);
    }

    /**
     * The shortcut-model loss on a batch: a share of flow matching at step size 0 and the rest
     * self-consistency between one step of size {@code 2d} and two of size {@code d}, with
     * {@code 2d} drawn from {@code 1, 1/2, ..., 1/2^(levels-1)}.
     *
     * @param net the average-velocity network
     * @param x0 data, {@code [batch, dimension]}
     * @param flowShare the share of the batch trained with plain flow matching, for example 0.75
     * @param levels how many step sizes
     * @param random the source of times, sizes and noise
     * @return the loss, a scalar
     */
    public static Var shortcutLoss(Network net, Tensor x0, float flowShare, int levels, RandomGenerator random) {
        int batch = x0.shape().dim(0);
        int flow = Math.max(1, Math.min(batch - 1, Math.round(batch * flowShare)));
        int self = batch - flow;
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random);
        float[] t = new float[batch];
        float[] r = new float[batch];
        float[] half = new float[self];
        for (int i = 0; i < batch; i++) {
            if (i < flow) {
                t[i] = (float) random.nextDouble();
                r[i] = t[i];
            } else {
                float size = (float) Math.pow(0.5, random.nextInt(levels));
                // t on the grid of this size, so the two half steps stay inside [0, 1]
                int slots = Math.round(1 / size);
                t[i] = size * (1 + random.nextInt(slots));
                r[i] = t[i] - size;
                half[i - flow] = size / 2;
            }
        }
        Tensor tc = column(t);
        Tensor zt = x0.mul(tc.neg().add(1f)).add(noise.mul(tc));
        Var u = net.apply(Var.of(zt), Var.of(column(r)), Var.of(tc));

        Var flowErr = u.slice(0, 0, flow).sub(Var.of(noise.sub(x0).slice(0, 0, flow))).square().mean();
        // two half steps without gradients give the target of the full step
        Tensor zs = zt.slice(0, flow, batch);
        Tensor ts = tc.slice(0, flow, batch);
        Tensor h = column(half);
        Tensor mid = ts.sub(h);
        Tensor u1 = net.apply(Var.of(zs), Var.of(mid), Var.of(ts)).value();
        Tensor zm = zs.sub(u1.mul(h));
        Tensor u2 = net.apply(Var.of(zm), Var.of(mid.sub(h)), Var.of(mid)).value();
        Tensor target = u1.add(u2).mul(0.5f);
        Var selfErr = u.slice(0, flow, batch).sub(Var.of(target)).square().mean();
        return flowErr.mul((float) flow / batch).add(selfErr.mul((float) self / batch));
    }

    /**
     * Generates from noise with equal average-velocity steps from time 1 to 0.
     *
     * @param net the trained network
     * @param noise standard normal starts, {@code [batch, dimension]}
     * @param steps the number of steps, 1 for one-step generation
     * @return the samples
     */
    public static Tensor sample(Network net, Tensor noise, int steps) {
        int batch = noise.shape().dim(0);
        Tensor x = noise;
        for (int i = 0; i < steps; i++) {
            float t = 1 - (float) i / steps;
            float r = 1 - (float) (i + 1) / steps;
            Tensor u = net.apply(Var.of(x), Var.of(Tensor.full(Shape.of(batch, 1), r)),
                    Var.of(Tensor.full(Shape.of(batch, 1), t))).value();
            x = x.sub(u.mul(t - r));
        }
        return x;
    }

    /**
     * The instantaneous velocity {@code u(x, t, t)} as a core {@link Denoiser}, for the core
     * multi-step samplers.
     *
     * @param net the network
     * @return the velocity field
     */
    public static Denoiser instantaneous(Network net) {
        return (noisy, time) -> {
            Tensor t = time.reshape(time.shape().dim(0), 1);
            return net.apply(noisy, Var.of(t), Var.of(t));
        };
    }

    private static Var weighted(Var err, float power) {
        if (power == 0) {
            return err.mean();
        }
        float[] e = err.value().toArray();
        float[] w = new float[e.length];
        for (int i = 0; i < e.length; i++) {
            w[i] = (float) (1 / Math.pow(e[i] + 1e-3, power));
        }
        return err.mul(Var.of(column(w))).mean();
    }

    private static Tensor column(float[] values) {
        return Tensor.of(Shape.of(values.length, 1), values.clone());
    }

    /** The default network: a residual MLP over the sample and time features. */
    private static final class IntervalMlp implements Network {
        private final int dimension;
        private final Tensor frequencies;
        private final Dense in;
        private final List<Dense> hidden = new ArrayList<>();
        private final Dense out;

        IntervalMlp(int dimension, int width, int depth, long seed) {
            SplittableRandom random = new SplittableRandom(seed);
            this.dimension = dimension;
            float[] f = new float[TIME_FEATURES / 2];
            for (int k = 0; k < f.length; k++) {
                f[k] = (float) (MAX_FREQUENCY * Math.exp(-Math.log(MAX_FREQUENCY) * k / Math.max(1, f.length - 1)));
            }
            frequencies = Tensor.of(Shape.of(1, f.length), f);
            in = GenBlocks.linear(dimension + 2 * TIME_FEATURES, width, random);
            for (int i = 0; i < depth; i++) {
                hidden.add(GenBlocks.linear(width, width, random));
            }
            out = GenBlocks.linear(width, dimension, random);
        }

        private Var features(Var time) {
            Var a = time.matmul(Var.of(frequencies));
            return Var.concat(1, a.sin(), a.cos());
        }

        @Override
        public Var apply(Var x, Var r, Var t) {
            Var h = GenBlocks.silu(in.forward(Var.concat(1, x, features(t), features(t.sub(r)))));
            for (Dense l : hidden) {
                h = h.add(GenBlocks.silu(l.forward(h)));
            }
            return out.forward(h);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(in.parameters());
            for (Dense l : hidden) {
                p.addAll(l.parameters());
            }
            p.addAll(out.parameters());
            return p;
        }

        @Override
        public int dimension() {
            return dimension;
        }
    }
}
