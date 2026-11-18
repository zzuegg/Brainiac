package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.IntrinsicReward;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Count-based exploration bonuses {@code 1 / sqrt(N(s'))} for large or continuous state
 * spaces, as {@link IntrinsicReward}s so any learner gets them through
 * {@code CuriosityEnvironment}:
 *
 * <ul>
 *   <li>{@link #tabular}: exact counts of distinct observations, the oracle where states are
 *       discrete;</li>
 *   <li>{@link #simHash} (#Exploration, Tang et al., NeurIPS 2017,
 *       <a href="https://arxiv.org/abs/1611.04717">arXiv:1611.04717</a>): states hashed to
 *       {@code k} bits by the signs of a fixed Gaussian projection {@code sign(A s)}, counted per
 *       code;</li>
 *   <li>{@link #learnedSimHash}: the same hash applied to the rounded code of an autoencoder
 *       with a sigmoid bottleneck trained online (uniform noise on the pre-activations and a
 *       binarisation penalty push the code to 0/1), for pixel inputs where raw projections hash
 *       irrelevant pixel noise;</li>
 *   <li>{@link #pseudoCount} (Bellemare et al., NeurIPS 2016,
 *       <a href="https://arxiv.org/abs/1606.01868">arXiv:1606.01868</a>): a density model
 *       (here a factorised categorical over binned dimensions) gives the probability
 *       {@code rho} of {@code s} before and {@code rho'} after one more update on it; the
 *       pseudo-count is {@code N = rho (1 - rho') / (rho' - rho)}.</li>
 * </ul>
 */
public final class CountBonuses implements IntrinsicReward {

    private enum Kind {
        TABULAR, SIMHASH, LEARNED_SIMHASH, PSEUDO_COUNT
    }

    private record Key(float[] v) {
        @Override
        public boolean equals(Object o) {
            return o instanceof Key k && Arrays.equals(v, k.v);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(v);
        }
    }

    private final Kind kind;
    private final int dims;
    private final Map<Object, Integer> counts = new HashMap<>();
    private double[][] projection;
    private Model encoder;
    private Model decoder;
    private Optimizer optimizer;
    private final List<float[]> recent = new ArrayList<>();
    private int codeSize;
    private long seen;
    private int bins;
    private float[] low;
    private float[] high;
    private int[][] binCounts;
    private final SplittableRandom random;

    private CountBonuses(Kind kind, int dims, long seed) {
        this.kind = kind;
        this.dims = dims;
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param observations values per observation
     * @return exact counts
     */
    public static CountBonuses tabular(int observations) {
        return new CountBonuses(Kind.TABULAR, observations, 0);
    }

    /**
     * @param observations values per observation
     * @param bits code bits {@code k}, at most 63
     * @param seed the projection seed
     * @return SimHash counts
     */
    public static CountBonuses simHash(int observations, int bits, long seed) {
        CountBonuses c = new CountBonuses(Kind.SIMHASH, observations, seed);
        c.projection = gaussian(bits, observations, c.random);
        return c;
    }

    /**
     * @param observations values per observation
     * @param code the autoencoder bottleneck width
     * @param bits hash bits applied to the rounded code
     * @param hidden the autoencoder hidden width
     * @param seed the seed
     * @return learned-code SimHash counts
     */
    public static CountBonuses learnedSimHash(int observations, int code, int bits, int hidden, long seed) {
        CountBonuses c = new CountBonuses(Kind.LEARNED_SIMHASH, observations, seed);
        c.codeSize = code;
        c.encoder = Nets.mlp(observations, code, Activation.RELU, seed, hidden);
        c.decoder = Nets.mlp(code, observations, Activation.RELU, seed + 1, hidden);
        c.optimizer = Optimizer.adam(1e-3f);
        c.projection = gaussian(bits, code, c.random);
        return c;
    }

    /**
     * @param low the lowest value of each dimension
     * @param high the highest value of each dimension
     * @param bins bins per dimension
     * @return pseudo-counts from a factorised categorical density
     */
    public static CountBonuses pseudoCount(float[] low, float[] high, int bins) {
        CountBonuses c = new CountBonuses(Kind.PSEUDO_COUNT, low.length, 0);
        c.low = low.clone();
        c.high = high.clone();
        c.bins = bins;
        c.binCounts = new int[low.length][bins];
        return c;
    }

    private static double[][] gaussian(int rows, int cols, SplittableRandom r) {
        double[][] a = new double[rows][cols];
        for (double[] row : a) {
            for (int j = 0; j < cols; j++) {
                row[j] = r.nextGaussian();
            }
        }
        return a;
    }

    private long hash(double[] x) {
        long code = 0;
        for (int i = 0; i < projection.length; i++) {
            double z = 0;
            for (int j = 0; j < x.length; j++) {
                z += projection[i][j] * x[j];
            }
            if (z > 0) {
                code |= 1L << i;
            }
        }
        return code;
    }

    private Object key(float[] obs) {
        return switch (kind) {
            case TABULAR -> new Key(obs.clone());
            case SIMHASH -> {
                double[] x = new double[obs.length];
                for (int i = 0; i < x.length; i++) {
                    x[i] = obs[i];
                }
                yield hash(x);
            }
            case LEARNED_SIMHASH -> {
                float[] b = RlTensors.row(encoder, obs);
                double[] x = new double[b.length];
                for (int i = 0; i < b.length; i++) {
                    x[i] = (1 / (1 + Math.exp(-b[i])) > 0.5 ? 1 : 0) - 0.5;
                }
                yield hash(x);
            }
            case PSEUDO_COUNT -> throw new IllegalStateException();
        };
    }

    private int bin(int d, float v) {
        double f = (v - low[d]) / (high[d] - low[d]);
        return (int) Math.max(0, Math.min(bins - 1, Math.floor(f * bins)));
    }

    /**
     * @param observation an observation
     * @return its current (pseudo-)count, without learning
     */
    public double count(float[] observation) {
        if (kind == Kind.PSEUDO_COUNT) {
            if (seen == 0) {
                return 0;
            }
            double logRatio = 0;
            double logRho = 0;
            for (int d = 0; d < dims; d++) {
                double c = binCounts[d][bin(d, observation[d])];
                double p = (c + 0.5) / (seen + 0.5 * bins);
                double p2 = (c + 1.5) / (seen + 1 + 0.5 * bins);
                logRho += Math.log(p);
                logRatio += Math.log(p2 / p);
            }
            double rho = Math.exp(logRho);
            double ratio = Math.expm1(logRatio);
            if (ratio <= 0) {
                return 1e9;
            }
            return Math.max(0, (1 - rho * (ratio + 1)) / ratio);
        }
        return counts.getOrDefault(key(observation), 0);
    }

    /** @return the number of distinct hash codes or states counted (0 for pseudo-counts) */
    public int distinct() {
        return counts.size();
    }

    @Override
    public float observe(float[] observation, float[] action, float[] next) {
        double n;
        if (kind == Kind.PSEUDO_COUNT) {
            n = count(next);
            for (int d = 0; d < dims; d++) {
                binCounts[d][bin(d, next[d])]++;
            }
            seen++;
        } else {
            if (kind == Kind.LEARNED_SIMHASH) {
                trainAutoencoder(next);
            }
            n = counts.merge(key(next), 1, Integer::sum);
        }
        return (float) (1 / Math.sqrt(n + 0.01));
    }

    private void trainAutoencoder(float[] obs) {
        recent.add(obs);
        if (recent.size() > 2_000) {
            recent.remove(random.nextInt(recent.size()));
        }
        if (++seen % 4 != 0 || recent.size() < 32) {
            return;
        }
        int b = 32;
        float[] x = new float[b * dims];
        float[] noise = new float[b * codeSize];
        for (int i = 0; i < b; i++) {
            System.arraycopy(recent.get(random.nextInt(recent.size())), 0, x, i * dims, dims);
        }
        for (int i = 0; i < noise.length; i++) {
            noise[i] = (float) random.nextDouble(-0.3, 0.3);
        }
        Tensor in = Tensor.of(Shape.of(b, dims), x);
        Var code = encoder.forward(Var.of(in), true).add(Tensor.of(Shape.of(b, codeSize), noise)).sigmoid();
        Var recon = decoder.forward(code, true);
        Var binary = code.mul(code.neg().add(1f)).mean();
        Var loss = recon.sub(in).square().mean().add(binary.mul(0.1f));
        loss.backward();
        List<Var> params = new ArrayList<>(encoder.parameters());
        params.addAll(decoder.parameters());
        optimizer.step(params);
    }

    @Override
    public String toString() {
        return kind.toString();
    }
}
