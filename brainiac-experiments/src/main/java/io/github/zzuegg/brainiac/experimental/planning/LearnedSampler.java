package io.github.zzuegg.brainiac.experimental.planning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A learned sampling distribution for sampling-based planners (Ichter, Harrison &amp; Pavone,
 * ICRA 2018, <a href="https://arxiv.org/abs/1709.05448">arXiv:1709.05448</a>): a conditional
 * variational autoencoder over configurations, conditioned on an encoding of the problem (a
 * coarse occupancy grid, start and goal), is trained on configurations taken from good paths
 * of earlier problems. At planning time a share {@code lambda} of the samples comes from the
 * decoder with {@code z ~ N(0, I)} and the rest stays uniform, which keeps probabilistic
 * completeness while concentrating samples where solutions lie (narrow passages).
 */
public final class LearnedSampler {

    private final SimpleMlp encoder;
    private final SimpleMlp decoder;
    private final int dims;
    private final int conditionSize;
    private final int latent;

    /**
     * Creates an untrained sampler.
     *
     * @param dims the configuration dimension
     * @param conditionSize the problem encoding size
     * @param latent the latent size
     * @param hidden the hidden width
     * @param seed the seed
     */
    public LearnedSampler(int dims, int conditionSize, int latent, int hidden, long seed) {
        this.dims = dims;
        this.conditionSize = conditionSize;
        this.latent = latent;
        this.encoder = SimpleMlp.relu(seed, dims + conditionSize, hidden, hidden, 2 * latent);
        this.decoder = SimpleMlp.relu(seed + 1, latent + conditionSize, hidden, hidden, dims);
    }

    /**
     * Trains by the conditional evidence lower bound.
     *
     * @param samples configurations from good paths
     * @param conditions the matching problem encodings
     * @param steps gradient steps
     * @param batch samples per step
     * @param beta the KL weight
     * @param learningRate Adam's rate
     * @param seed the seed
     * @return the last loss
     */
    public double fit(List<double[]> samples, List<double[]> conditions, int steps, int batch, float beta,
            float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> params = new java.util.ArrayList<>(encoder.parameters());
        params.addAll(decoder.parameters());
        double last = 0;
        for (int s = 0; s < steps; s++) {
            float[] xc = new float[batch * (dims + conditionSize)];
            float[] x = new float[batch * dims];
            float[] c = new float[batch * conditionSize];
            float[] eps = new float[batch * latent];
            for (int i = 0; i < batch; i++) {
                int k = r.nextInt(samples.size());
                for (int j = 0; j < dims; j++) {
                    x[i * dims + j] = (float) samples.get(k)[j];
                    xc[i * (dims + conditionSize) + j] = x[i * dims + j];
                }
                for (int j = 0; j < conditionSize; j++) {
                    c[i * conditionSize + j] = (float) conditions.get(k)[j];
                    xc[i * (dims + conditionSize) + dims + j] = c[i * conditionSize + j];
                }
                for (int j = 0; j < latent; j++) {
                    eps[i * latent + j] = (float) r.nextGaussian();
                }
            }
            Var stats = encoder.forward(Var.of(Tensor.of(Shape.of(batch, dims + conditionSize), xc)));
            Var mu = stats.slice(1, 0, latent);
            Var logVar = stats.slice(1, latent, 2 * latent).clamp(-6f, 4f);
            Var z = mu.add(logVar.mul(0.5f).exp().mul(Tensor.of(Shape.of(batch, latent), eps)));
            Var recon = decoder.forward(Var.concat(1, z, Var.of(Tensor.of(Shape.of(batch, conditionSize), c))));
            Var rec = recon.sub(Var.of(Tensor.of(Shape.of(batch, dims), x))).square().sum(1, false).mean();
            Var kl = logVar.exp().add(mu.square()).sub(logVar).sub(1f).sum(1, false).mean().mul(0.5f);
            Var loss = rec.add(kl.mul(beta));
            last = loss.value().item();
            loss.backward();
            opt.step(params);
        }
        return last;
    }

    /**
     * Draws configurations for one problem.
     *
     * @param condition the problem encoding
     * @param count how many
     * @param random the random source
     * @return {@code [count][dims]}
     */
    public double[][] generate(double[] condition, int count, SplittableRandom random) {
        float[] in = new float[count * (latent + conditionSize)];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < latent; j++) {
                in[i * (latent + conditionSize) + j] = (float) random.nextGaussian();
            }
            for (int j = 0; j < conditionSize; j++) {
                in[i * (latent + conditionSize) + latent + j] = (float) condition[j];
            }
        }
        float[] out = decoder.predict(Tensor.of(Shape.of(count, latent + conditionSize), in)).toArray();
        double[][] q = new double[count][dims];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dims; j++) {
                q[i][j] = out[i * dims + j];
            }
        }
        return q;
    }

    /**
     * A planner sampler mixing learned and uniform samples; learned samples are pre-drawn in
     * blocks of 256 for speed.
     *
     * @param condition the problem encoding
     * @param uniform the uniform sampler
     * @param lambda the share of learned samples
     * @return the sampler
     */
    public SamplingPlanners.Sampler sampler(double[] condition, SamplingPlanners.Sampler uniform, double lambda) {
        return new SamplingPlanners.Sampler() {
            private double[][] pool;
            private int next;

            @Override
            public double[] sample(SplittableRandom random) {
                if (random.nextDouble() >= lambda) {
                    return uniform.sample(random);
                }
                if (pool == null || next >= pool.length) {
                    pool = generate(condition, 256, random);
                    next = 0;
                }
                return pool[next++];
            }
        };
    }
}
