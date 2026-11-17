package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MeanFlow;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Continuous-time consistency models, sCM (Lu and Song,
 * <a href="https://arxiv.org/abs/2410.11081">arXiv:2410.11081</a>), trained by consistency
 * training (no teacher). TrigFlow puts data and noise on a circle,
 * {@code x_t = cos(t) x0 + sin(t) z} with {@code t in [0, pi/2]} and {@code sigma_d = 1}, and the
 * consistency function is {@code f(x_t, t) = cos(t) x_t - sin(t) F(x_t, t)}, so {@code f(x, 0) = x}
 * holds by construction. Discrete consistency models compare {@code f} at two nearby times;
 * the continuous objective takes the limit: the training target moves {@code F} along
 * {@code cos(t) df/dt}, the exact time derivative of the stop-gradient {@code f} along the
 * trajectory, with {@code dF/dt} computed as a Jacobian-vector product (two reverse passes,
 * as in {@link MeanFlow}). The paper's stabilisers are included: tangent normalisation
 * {@code g / (|g| + c)} and a warmup that ramps in the {@code sin(t) dF/dt} term.
 *
 * <p>The network is a {@link MeanFlow.Network} called as {@code F(x, t, t)}: it is built from
 * twice-differentiable operations and takes time as a differentiable input.
 */
public final class ContinuousConsistency {
    private static final float HALF_PI = (float) (Math.PI / 2);
    private final MeanFlow.Network net;
    private final float meanLogSigma;
    private final float stdLogSigma;
    private final int warmup;
    private int steps;

    private ContinuousConsistency(MeanFlow.Network net, float meanLogSigma, float stdLogSigma, int warmup) {
        this.net = net;
        this.meanLogSigma = meanLogSigma;
        this.stdLogSigma = stdLogSigma;
        this.warmup = warmup;
    }

    /**
     * Creates the trainer.
     *
     * @param net the network {@code F}
     * @param meanLogSigma the mean of {@code ln tan t}, -1 in the paper
     * @param stdLogSigma its standard deviation, 1.4 in the paper
     * @param warmup training steps over which the {@code sin(t) dF/dt} term ramps in
     * @return the trainer
     */
    public static ContinuousConsistency of(MeanFlow.Network net, float meanLogSigma, float stdLogSigma, int warmup) {
        return new ContinuousConsistency(net, meanLogSigma, stdLogSigma, warmup);
    }

    /** @return the network */
    public MeanFlow.Network network() {
        return net;
    }

    private Var f(Var x, Var t) {
        return net.apply(x, t, t);
    }

    /**
     * The sCT loss on a batch of data with unit-ish scale; advances the warmup counter.
     *
     * @param x0 data {@code [batch, dim]}
     * @param random the source of times and noise
     * @return the scalar loss
     */
    public Var loss(Tensor x0, RandomGenerator random) {
        int b = x0.shape().dim(0);
        int dim = x0.shape().dim(1);
        float[] t = new float[b];
        float[] cos = new float[b];
        float[] sin = new float[b];
        for (int i = 0; i < b; i++) {
            t[i] = (float) Math.atan(Math.exp(meanLogSigma + stdLogSigma * random.nextGaussian()));
            cos[i] = (float) Math.cos(t[i]);
            sin[i] = (float) Math.sin(t[i]);
        }
        Tensor c = GenerativeOps.column(cos);
        Tensor s = GenerativeOps.column(sin);
        Tensor z = Tensor.normal(x0.shape(), 0, 1, random);
        Tensor xt = x0.mul(c).add(z.mul(s));
        Tensor dxdt = x0.mul(s).neg().add(z.mul(c));

        // F and dF/dt along (dx/dt, 1) by the double-VJP trick
        Var xv = Var.param(xt);
        Var tv = Var.param(GenerativeOps.column(t));
        Var fMinus = f(xv, tv);
        Var w = Var.param(Tensor.zeros(Shape.of(b, dim)));
        List<Var> vjp = Var.gradients(fMinus, w, List.of(xv, tv), true);
        Var directional = vjp.get(0).mul(Var.of(dxdt)).sum().add(vjp.get(1).sum());
        Tensor dFdt = Var.gradients(directional, List.of(w), false).get(0).value();
        Tensor fm = fMinus.value();

        float ramp = warmup <= 0 ? 1 : Math.min(1, (float) steps / warmup);
        steps++;
        // g = cos(t) df/dt = -cos^2 (F - dx/dt) - r cos sin (x_t + dF/dt)
        Tensor g = fm.sub(dxdt).mul(c.square()).neg()
                .sub(xt.add(dFdt).mul(c.mul(s)).mul(ramp));
        float[] gv = g.toArray();
        float[] scale = new float[b];
        for (int i = 0; i < b; i++) {
            double norm = 0;
            for (int k = 0; k < dim; k++) {
                norm += gv[i * dim + k] * gv[i * dim + k];
            }
            scale[i] = (float) (1 / (Math.sqrt(norm / dim) + 0.1));
        }
        Tensor target = fm.add(g.mul(GenerativeOps.column(scale)));
        Var out = f(Var.of(xt), Var.of(GenerativeOps.column(t)));
        return out.sub(Var.of(target)).square().mean();
    }

    /**
     * The consistency function's clean estimate {@code cos(t) x - sin(t) F(x, t)}.
     *
     * @param x points at time {@code t}
     * @param t the TrigFlow time
     * @return the estimate of {@code x0}
     */
    public Tensor denoise(Tensor x, float t) {
        int n = x.shape().dim(0);
        Tensor tc = Tensor.full(Shape.of(n, 1), t);
        Tensor fv = f(Var.of(x), Var.of(tc)).value();
        return x.mul((float) Math.cos(t)).sub(fv.mul((float) Math.sin(t)));
    }

    /**
     * One- or few-step sampling: denoise from {@code t = pi/2}, then (for more steps) re-noise
     * to the next intermediate time and denoise again.
     *
     * @param noise standard normal noise {@code [n, dim]}
     * @param times intermediate TrigFlow times after the first step, descending (empty for one step)
     * @param random the source of the re-noising
     * @return the samples
     */
    public Tensor sample(Tensor noise, float[] times, RandomGenerator random) {
        Tensor x = denoise(noise, HALF_PI);
        for (float t : times) {
            Tensor z = Tensor.normal(x.shape(), 0, 1, random);
            x = denoise(x.mul((float) Math.cos(t)).add(z.mul((float) Math.sin(t))), t);
        }
        return x;
    }
}
