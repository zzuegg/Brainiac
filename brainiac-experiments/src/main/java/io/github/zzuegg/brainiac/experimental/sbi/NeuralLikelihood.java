package io.github.zzuegg.brainiac.experimental.sbi;

import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;

import java.util.SplittableRandom;

/**
 * Neural likelihood estimation (Papamakarios, Sterratt and Murray, "Sequential neural
 * likelihood", AISTATS 2019, <a href="https://arxiv.org/abs/1805.07226">arXiv:1805.07226</a>):
 * learn a conditional density {@code q(x | theta)} of the data given the parameters (a
 * {@link ConditionalDensity}, mixture density network or coupling flow), then sample
 * {@code p(theta | x_o) ~ q(x_o | theta) p(theta)} by {@link SliceSampler slice sampling}.
 * Unlike posterior estimation it reuses one learned likelihood for any number of i.i.d.
 * observations by summing their log likelihoods. {@link #sequential} (SNL) retrains on
 * simulations from the current posterior; the likelihood is valid whatever proposal produced
 * the training parameters, so no correction is needed.
 */
public final class NeuralLikelihood {

    private final ConditionalDensity density;
    private final long simulations;

    private NeuralLikelihood(ConditionalDensity density, long simulations) {
        this.density = density;
        this.simulations = simulations;
    }

    /**
     * Trains on prior simulations.
     *
     * @param prior the prior
     * @param sim the simulator
     * @param simulations the simulation budget
     * @param kind the density family
     * @param config training settings
     * @param seed the seed
     * @return the likelihood estimator
     */
    public static NeuralLikelihood train(Prior prior, Simulator sim, int simulations, ConditionalDensity.Kind kind,
            ConditionalDensity.Config config, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] t = new double[simulations][];
        double[][] x = new double[simulations][];
        for (int i = 0; i < simulations; i++) {
            t[i] = prior.sample(r);
            x[i] = sim.simulate(t[i], r);
        }
        return new NeuralLikelihood(ConditionalDensity.fit(kind, x, t, config, seed), simulations);
    }

    /**
     * Sequential neural likelihood: round one simulates from the prior, each later round from
     * the current posterior given {@code observations}, and the density is retrained on all
     * simulations so far.
     *
     * @param prior the prior
     * @param sim the simulator
     * @param observations the i.i.d. observations {@code [k][dx]}
     * @param rounds the number of rounds
     * @param perRound simulations per round
     * @param kind the density family
     * @param config training settings (iterations per round)
     * @param seed the seed
     * @return the final estimator
     */
    public static NeuralLikelihood sequential(Prior prior, Simulator sim, double[][] observations, int rounds,
            int perRound, ConditionalDensity.Kind kind, ConditionalDensity.Config config, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] t = new double[rounds * perRound][];
        double[][] x = new double[rounds * perRound][];
        NeuralLikelihood nl = null;
        int n = 0;
        for (int round = 0; round < rounds; round++) {
            double[][] proposal = round == 0 ? null : nl.posterior(prior, observations, perRound, r);
            for (int i = 0; i < perRound; i++) {
                t[n] = round == 0 ? prior.sample(r) : proposal[i];
                x[n] = sim.simulate(t[n], r);
                n++;
            }
            double[][] tt = java.util.Arrays.copyOf(t, n);
            double[][] xx = java.util.Arrays.copyOf(x, n);
            if (nl == null) {
                nl = new NeuralLikelihood(ConditionalDensity.fit(kind, xx, tt, config, seed), n);
            } else {
                nl.density.train(xx, tt, config.iterations(), seed + round);
                nl = new NeuralLikelihood(nl.density, n);
            }
        }
        return nl;
    }

    /**
     * The summed log likelihood of i.i.d. observations.
     *
     * @param theta the parameters
     * @param observations {@code [k][dx]}
     * @return {@code sum_i log q(x_i | theta)}
     */
    public double logLikelihood(double[] theta, double[][] observations) {
        double[][] c = new double[observations.length][];
        java.util.Arrays.fill(c, theta);
        double s = 0;
        for (double v : density.logDensities(observations, c)) {
            s += v;
        }
        return s;
    }

    /**
     * Posterior samples by slice sampling the learned likelihood times the prior, with four
     * chains started at the best of 500 prior draws.
     *
     * @param prior the prior
     * @param observations the i.i.d. observations
     * @param n the number of samples
     * @param random the random source
     * @return {@code [n][d]}
     */
    public double[][] posterior(Prior prior, double[][] observations, int n, SplittableRandom random) {
        double[][] cand = new double[500][];
        for (int i = 0; i < cand.length; i++) {
            cand[i] = prior.sample(random);
        }
        return SliceSampler.sample(th -> {
            double lp = prior.logDensity(th);
            return Double.isFinite(lp) ? lp + logLikelihood(th, observations) : lp;
        }, cand, 4, 60, n, 0.5, random);
    }

    /**
     * The learned conditional density {@code q(x | theta)}.
     *
     * @return the density
     */
    public ConditionalDensity density() {
        return density;
    }

    /**
     * The simulations used.
     *
     * @return the count
     */
    public long simulations() {
        return simulations;
    }
}
