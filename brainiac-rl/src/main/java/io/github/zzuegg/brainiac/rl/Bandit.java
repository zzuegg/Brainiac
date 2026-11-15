package io.github.zzuegg.brainiac.rl;

import java.util.SplittableRandom;

/**
 * A multi-armed bandit: every episode is a single pull of one of {@code arms} levers.
 * Each arm pays a reward drawn from a normal distribution around its own mean; the means
 * are fixed by the seed. The observation is always {@code [1]}.
 */
public final class Bandit implements Environment {

    private final double[] means;
    private final double noise;
    private SplittableRandom random = new SplittableRandom(0);
    private boolean over = true;

    /**
     * @param arms the number of arms
     * @param seed picks the arm means, from {@code N(0, 1)}
     */
    public Bandit(int arms, long seed) {
        this(randomMeans(arms, seed), 1);
    }

    /**
     * @param means the mean reward of every arm
     * @param noise the standard deviation of rewards
     */
    public Bandit(double[] means, double noise) {
        if (means.length < 1) {
            throw new IllegalArgumentException("a bandit needs at least one arm");
        }
        this.means = means.clone();
        this.noise = noise;
    }

    private static double[] randomMeans(int arms, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] m = new double[arms];
        for (int i = 0; i < arms; i++) {
            m[i] = r.nextGaussian();
        }
        return m;
    }

    /** @return the index of the arm with the highest mean */
    public int bestArm() {
        int best = 0;
        for (int i = 1; i < means.length; i++) {
            if (means[i] > means[best]) {
                best = i;
            }
        }
        return best;
    }

    /** @return the mean reward of every arm */
    public double[] means() {
        return means.clone();
    }

    @Override
    public Space observationSpace() {
        return Space.box(1, 1, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(means.length);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        this.random = random;
        over = false;
        return new float[] {1};
    }

    @Override
    public Step step(float[] action) {
        if (over) {
            throw new IllegalStateException("the episode is over; call reset");
        }
        int a = Math.round(action[0]);
        over = true;
        return new Step(new float[] {1}, (float) (means[a] + noise * random.nextGaussian()), true, false);
    }

    @Override
    public String toString() {
        return "Bandit(" + means.length + " arms)";
    }
}
