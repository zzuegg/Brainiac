package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A latent SDE (Li, Wong, Chen and Duvenaud, 2020) for irregularly sampled stochastic series.
 * A latent state follows a prior SDE {@code dz = f_theta(z) dt + g dW}; an amortised posterior
 * SDE {@code dz = h_phi(z, context_t) dt + g dW}, whose drift sees the next observation and
 * the time until it, explains the data; observations are {@code y = W z + b + noise}. Both
 * SDEs share the diffusion, so the KL divergence between their path measures is the pathwise
 * integral {@code int 1/2 |(h - f) / g|^2 dt} (Girsanov), and training maximises the ELBO
 *
 * <pre>
 * sum_obs log p(y | z_t) - KL(q(z0) || p(z0)) - int 1/2 |(h - f)/g|^2 dt
 * </pre>
 *
 * <p>with posterior paths sampled by a differentiable Euler-Maruyama solver
 * ({@link SdeSolver#eulerMaruyama}) and gradients taken directly through it. Sampling the
 * prior SDE generates new series with calibrated spread.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2001.01328">arXiv:2001.01328</a>.
 */
public final class LatentSde {

    private final int latent;
    private final SimpleMlp prior;
    private final SimpleMlp posterior;
    private final SimpleMlp initial;
    private final Var logDiffusion;
    private final Var priorMean;
    private final Var priorLogStd;
    private final Var readout;
    private final Var readoutBias;
    private final Var logNoise;

    private LatentSde(int latent, int width, long seed) {
        this.latent = latent;
        this.prior = SimpleMlp.tanh(seed, latent, width, latent);
        this.posterior = SimpleMlp.tanh(seed + 1, latent + 2, width, latent);
        this.initial = SimpleMlp.tanh(seed + 2, 1, width, 2 * latent);
        this.logDiffusion = Var.param(Tensor.full(Shape.of(1, latent), -1f));
        this.priorMean = Var.param(Tensor.zeros(Shape.of(1, latent)));
        this.priorLogStd = Var.param(Tensor.zeros(Shape.of(1, latent)));
        SplittableRandom r = new SplittableRandom(seed + 3);
        this.readout = Var.param(Tensor.normal(Shape.of(latent, 1), 0, 1f / latent, r));
        this.readoutBias = Var.param(Tensor.zeros(Shape.of(1, 1)));
        this.logNoise = Var.param(Tensor.full(Shape.of(1, 1), -1f));
    }

    /**
     * Creates the model for scalar observations.
     *
     * @param latent the latent size
     * @param width the hidden width
     * @param seed the seed
     * @return the model
     */
    public static LatentSde of(int latent, int width, long seed) {
        return new LatentSde(latent, width, seed);
    }

    /**
     * A batch of irregularly sampled scalar series on a common time grid.
     *
     * @param values observed values {@code [b][steps + 1]} (ignored where not observed)
     * @param observed whether each grid time is observed {@code [b][steps + 1]}; time 0 must be
     */
    public record Batch(double[][] values, boolean[][] observed) {
    }

    private Tensor[] contexts(Batch batch, float dt) {
        int b = batch.values().length;
        int t = batch.values()[0].length;
        Tensor[] ctx = new Tensor[t];
        for (int s = 0; s < t; s++) {
            float[] c = new float[b * 2];
            for (int i = 0; i < b; i++) {
                int next = -1;
                for (int k = s + 1; k < t; k++) {
                    if (batch.observed()[i][k]) {
                        next = k;
                        break;
                    }
                }
                c[2 * i] = next < 0 ? 0f : (float) batch.values()[i][next];
                c[2 * i + 1] = next < 0 ? 0f : (next - s) * dt;
            }
            ctx[s] = Tensor.of(Shape.of(b, 2), c);
        }
        return ctx;
    }

    /**
     * The negative ELBO per observation of a batch.
     *
     * @param batch the series
     * @param dt the grid step
     * @param random the source of the posterior noise
     * @return the loss
     */
    public Var loss(Batch batch, float dt, SplittableRandom random) {
        int b = batch.values().length;
        int steps = batch.values()[0].length - 1;
        Tensor[] ctx = contexts(batch, dt);
        float[] first = new float[b];
        for (int i = 0; i < b; i++) {
            first[i] = (float) batch.values()[i][0];
        }
        Var enc = initial.forward(Var.of(Tensor.of(Shape.of(b, 1), first)));
        Var mu = enc.slice(1, 0, latent);
        Var logStd = enc.slice(1, latent, 2 * latent).sub(1f);
        Var z = mu.add(logStd.exp().mul(Var.of(Tensor.normal(Shape.of(b, latent), 0, 1, random))));
        // KL(N(mu, s) || N(m, sp)) per example
        Var sp = priorLogStd.exp();
        Var kl0 = priorLogStd.sub(logStd).add(logStd.exp().square().add(mu.sub(priorMean).square())
                .div(sp.square().mul(2f))).sub(0.5f).sum();
        Var g = logDiffusion.exp();
        Var noise = logNoise.exp();
        Var logLik = observationLogLik(z, batch, 0, noise);
        Var pathKl = null;
        float sq = (float) Math.sqrt(dt);
        for (int s = 0; s < steps; s++) {
            Var zc = z;
            Var h = posterior.forward(Var.concat(1, zc, Var.of(ctx[s])));
            Var f = prior.forward(zc);
            Var u = h.sub(f).div(g);
            Var term = u.square().sum().mul(0.5f * dt);
            pathKl = pathKl == null ? term : pathKl.add(term);
            Tensor dw = Tensor.normal(Shape.of(b, latent), 0, sq, random);
            z = zc.add(h.mul(dt)).add(g.mul(Var.of(dw)));
            logLik = logLik.add(observationLogLik(z, batch, s + 1, noise));
        }
        int count = 0;
        for (boolean[] row : batch.observed()) {
            for (boolean o : row) {
                count += o ? 1 : 0;
            }
        }
        return kl0.add(pathKl).sub(logLik).div(count);
    }

    private Var observationLogLik(Var z, Batch batch, int s, Var noise) {
        int b = batch.values().length;
        float[] y = new float[b];
        float[] m = new float[b];
        for (int i = 0; i < b; i++) {
            if (batch.observed()[i][s]) {
                y[i] = (float) batch.values()[i][s];
                m[i] = 1;
            }
        }
        Var pred = z.matmul(readout).add(readoutBias);
        Var r = pred.sub(Var.of(Tensor.of(Shape.of(b, 1), y))).div(noise);
        Var ll = r.square().mul(-0.5f).sub(noise.log()).sub((float) (0.5 * Math.log(2 * Math.PI)));
        return ll.mul(Var.of(Tensor.of(Shape.of(b, 1), m))).sum();
    }

    /** @return every trainable parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(prior.parameters());
        p.addAll(posterior.parameters());
        p.addAll(initial.parameters());
        p.addAll(List.of(logDiffusion, priorMean, priorLogStd, readout, readoutBias, logNoise));
        return p;
    }

    /**
     * Generates series from the prior SDE.
     *
     * @param count the series
     * @param dt the step
     * @param steps the steps
     * @param random the source
     * @return observations with noise {@code [count][steps + 1]}
     */
    public double[][] sample(int count, double dt, int steps, SplittableRandom random) {
        MlpSnapshot f = new MlpSnapshot(prior);
        float[] g = logDiffusion.value().exp().toArray();
        float[] pm = priorMean.value().toArray();
        float[] ps = priorLogStd.value().exp().toArray();
        float[] w = readout.value().toArray();
        double bias = readoutBias.value().item();
        double noise = Math.exp(logNoise.value().item());
        double[][] out = new double[count][steps + 1];
        double sq = Math.sqrt(dt);
        for (int i = 0; i < count; i++) {
            double[] z = new double[latent];
            for (int k = 0; k < latent; k++) {
                z[k] = pm[k] + ps[k] * random.nextGaussian();
            }
            for (int s = 0; s <= steps; s++) {
                if (s > 0) {
                    double[] drift = f.forward(z);
                    for (int k = 0; k < latent; k++) {
                        z[k] += drift[k] * dt + g[k] * sq * random.nextGaussian();
                    }
                }
                double y = bias;
                for (int k = 0; k < latent; k++) {
                    y += w[k] * z[k];
                }
                out[i][s] = y + noise * random.nextGaussian();
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return "LatentSde(latent=" + latent + ")";
    }
}
