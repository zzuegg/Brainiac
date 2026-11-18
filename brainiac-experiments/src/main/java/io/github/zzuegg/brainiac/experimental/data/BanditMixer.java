package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Online data mixing as a multi-armed bandit (Albalak et al., "Efficient Online Data Mixing For
 * Language Model Pre-Training", 2023, <a href="https://arxiv.org/abs/2312.02406">arXiv:2312.02406</a>).
 * Each domain is an arm of an EXP3 bandit; the reward of sampling a domain is its training loss
 * (high loss means more to learn), importance-weighted by the probability of sampling it and
 * smoothed by a moving average. The mixture is set during the one real training run, with no
 * reference model and no proxy runs (unlike {@link DataMixture#updateWithExcessLoss DoReMi}):
 * <pre>
 *   ε_t = min(1/K, √(ln K / (K t)))
 *   π   = (1 - K ε_t) softmax(ε_{t-1} R̂) + ε_t
 *   R̂_i ← α R̂_i + (1 - α) L_i / π_i      for every domain in the batch
 * </pre>
 * A domain whose loss stays high because it is noise attracts weight it cannot use; with
 * {@link #clipToMedian} the loss reward of a domain is capped at the median of the latest
 * domain losses, which bounds that pull.
 */
public final class BanditMixer {

    private final int domains;
    private final double alpha;
    private final boolean clip;
    private final double[] reward;
    private final double[] lastLoss;
    private double[] weights;
    private long t;
    private double previousEpsilon;

    private BanditMixer(int domains, double alpha, boolean clip) {
        this.domains = domains;
        this.alpha = alpha;
        this.clip = clip;
        this.reward = new double[domains];
        this.lastLoss = new double[domains];
        Arrays.fill(lastLoss, Double.NaN);
        this.weights = new double[domains];
        Arrays.fill(weights, 1.0 / domains);
        this.previousEpsilon = 1.0 / domains;
    }

    /**
     * The mixer of Albalak et al.
     *
     * @param domains the number of domains
     * @param alpha the reward smoothing, for example 0.9
     * @return the mixer, starting uniform
     */
    public static BanditMixer of(int domains, double alpha) {
        return new BanditMixer(domains, alpha, false);
    }

    /**
     * The mixer with each domain's loss reward capped at the median of the latest losses.
     *
     * @param domains the number of domains
     * @param alpha the reward smoothing
     * @return the mixer
     */
    public static BanditMixer clipToMedian(int domains, double alpha) {
        return new BanditMixer(domains, alpha, true);
    }

    /** @return the current mixture */
    public double[] weights() {
        return weights.clone();
    }

    /** @return the current mixture as a constant {@link DataMixture} */
    public DataMixture mixture() {
        return DataMixture.constant(weights);
    }

    /**
     * Samples a domain from the current mixture.
     *
     * @param random the random source
     * @return the domain
     */
    public int sample(RandomGenerator random) {
        double u = random.nextDouble();
        for (int i = 0; i < domains - 1; i++) {
            u -= weights[i];
            if (u < 0) {
                return i;
            }
        }
        return domains - 1;
    }

    /**
     * Updates the rewards with the losses of the domains in the last batch and recomputes the
     * mixture.
     *
     * @param losses each domain's mean training loss in the batch (ignored where not pulled)
     * @param pulled which domains were in the batch
     * @return the new mixture
     */
    public double[] update(double[] losses, boolean[] pulled) {
        t++;
        for (int i = 0; i < domains; i++) {
            if (pulled[i]) {
                lastLoss[i] = losses[i];
            }
        }
        double cap = Double.POSITIVE_INFINITY;
        if (clip) {
            double[] seen = Arrays.stream(lastLoss).filter(Double::isFinite).sorted().toArray();
            if (seen.length > 0) {
                cap = seen.length % 2 == 1 ? seen[seen.length / 2]
                        : 0.5 * (seen[seen.length / 2 - 1] + seen[seen.length / 2]);
            }
        }
        for (int i = 0; i < domains; i++) {
            if (pulled[i]) {
                reward[i] = alpha * reward[i] + (1 - alpha) * Math.min(losses[i], cap) / weights[i];
            }
        }
        double eps = Math.min(1.0 / domains, Math.sqrt(Math.log(domains) / (domains * (double) t)));
        double max = Double.NEGATIVE_INFINITY;
        for (double r : reward) {
            max = Math.max(max, previousEpsilon * r);
        }
        double sum = 0;
        double[] w = new double[domains];
        for (int i = 0; i < domains; i++) {
            w[i] = Math.exp(previousEpsilon * reward[i] - max);
            sum += w[i];
        }
        for (int i = 0; i < domains; i++) {
            w[i] = (1 - domains * eps) * w[i] / sum + eps;
        }
        weights = w;
        previousEpsilon = eps;
        return weights.clone();
    }

    /**
     * Updates after a batch drawn from one domain.
     *
     * @param domain the domain pulled
     * @param loss its loss
     * @return the new mixture
     */
    public double[] update(int domain, double loss) {
        double[] l = new double[domains];
        boolean[] p = new boolean[domains];
        l[domain] = loss;
        p[domain] = true;
        return update(l, p);
    }

    @Override
    public String toString() {
        return "BanditMixer" + Arrays.toString(weights);
    }
}
