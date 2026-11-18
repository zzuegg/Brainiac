package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * MixStyle (Zhou et al., ICLR 2021, <a href="https://arxiv.org/abs/2104.02008">arXiv:2104.02008</a>):
 * inside an early CNN layer, the per-instance channel means and standard deviations (the
 * "style") of each example are mixed with those of another example,
 * {@code mu' = l mu + (1 - l) mu[perm]}, {@code sigma'} likewise with
 * {@code l ~ Beta(alpha, alpha)}, and the normalised content is restyled with them. Mixing
 * across training domains synthesises new styles for free, which helps domain generalisation.
 * The statistics are detached, as in the paper.
 *
 * <p>It acts only in training mode, with probability {@code p} per batch. The partner of each
 * example is random, or, after {@link #domains}, drawn from another domain where possible
 * (the paper's domain-aware shuffle). Input {@code [n, channels, ...spatial]}.
 */
public final class MixStyle implements Layer {

    private final double probability;
    private final double alpha;
    private final SplittableRandom random;
    private int[] domains;

    /**
     * Creates the layer.
     *
     * @param probability the chance of mixing a batch, 0.5 in the paper
     * @param alpha the Beta concentration, 0.1 in the paper
     * @param seed the seed of the mixing randomness
     */
    public MixStyle(double probability, double alpha, long seed) {
        this.probability = probability;
        this.alpha = alpha;
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param probability the chance of mixing a batch
     * @param alpha the Beta concentration
     * @param seed the seed
     * @return a spec for {@code Model.sequential().add(...)}
     */
    public static LayerSpec spec(double probability, double alpha, long seed) {
        return LayerSpec.of(new MixStyle(probability, alpha, seed));
    }

    /**
     * Sets the domains of the next training batch, for domain-aware shuffling.
     *
     * @param domainOfRow the domain of each row, or {@code null} for random partners
     */
    public void domains(int[] domainOfRow) {
        this.domains = domainOfRow == null ? null : domainOfRow.clone();
    }

    @Override
    public Var forward(Var input) {
        return input;
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (!training || random.nextDouble() >= probability) {
            return input;
        }
        Shape s = input.shape();
        int n = s.dim(0);
        int c = s.dim(1);
        int spatial = Math.toIntExact(s.size() / ((long) n * c));
        Var flat = input.reshape(n, c, spatial);
        Tensor mu = flat.value().mean(2, true);
        Tensor sig = flat.value().variance(2, 0, true).add(1e-6f).sqrt();
        Var normed = flat.sub(mu).div(sig);
        int[] perm = partners(n);
        float[] lam = new float[n];
        for (int i = 0; i < n; i++) {
            lam[i] = (float) beta();
        }
        Tensor l = Tensor.of(Shape.of(n, 1, 1), lam);
        Tensor one = Tensor.full(Shape.of(n, 1, 1), 1f);
        Tensor idx = Tensor.of(Shape.of(n), toFloat(perm));
        Tensor muMix = mu.mul(l).add(mu.gather(0, idx).mul(one.sub(l)));
        Tensor sigMix = sig.mul(l).add(sig.gather(0, idx).mul(one.sub(l)));
        domains = null;
        return normed.mul(sigMix).add(muMix).reshape(s.toArray());
    }

    private int[] partners(int n) {
        int[] perm = new int[n];
        if (domains != null && domains.length == n) {
            for (int i = 0; i < n; i++) {
                int pick = i;
                for (int tries = 0; tries < 8; tries++) {
                    int j = random.nextInt(n);
                    pick = j;
                    if (domains[j] != domains[i]) {
                        break;
                    }
                }
                perm[i] = pick;
            }
            return perm;
        }
        for (int i = 0; i < n; i++) {
            perm[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        return perm;
    }

    private double beta() {
        double a = gamma(alpha);
        double b = gamma(alpha);
        return a / (a + b + 1e-300);
    }

    private double gamma(double shape) {
        if (shape < 1) {
            return gamma(shape + 1) * Math.pow(random.nextDouble(), 1 / shape);
        }
        double d = shape - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double x = random.nextGaussian();
            double v = 1 + c * x;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            if (Math.log(random.nextDouble()) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
                return d * v;
            }
        }
    }

    private static float[] toFloat(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return f;
    }
}
