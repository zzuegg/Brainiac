package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A hidden Markov model with discrete or Gaussian emissions (Rabiner 1989,
 * <a href="https://ieeexplore.ieee.org/document/18626">A tutorial on hidden Markov models and
 * selected applications in speech recognition</a>): the scaled forward-backward recursions for
 * the likelihood and the state posteriors, Viterbi decoding of the most likely state path, and
 * Baum-Welch (EM) training from unlabelled sequences.
 *
 * <p>Observations are {@code double}s; for discrete emissions they hold symbol indices. The
 * static {@link #viterbi(double[], double[][], double[][])} works on any log-space chain
 * scores and is shared with the linear-chain CRF.
 */
public final class HiddenMarkovModel {

    private final double[] start;
    private final double[][] transition;
    private final double[][] symbols;      // discrete: [states][symbols], else null
    private final double[] means;          // Gaussian, else null
    private final double[] variances;

    private HiddenMarkovModel(double[] start, double[][] transition, double[][] symbols, double[] means,
            double[] variances) {
        this.start = start;
        this.transition = transition;
        this.symbols = symbols;
        this.means = means;
        this.variances = variances;
    }

    /**
     * A model with discrete emissions.
     *
     * @param start the initial state distribution
     * @param transition {@code [from][to]} probabilities
     * @param emission {@code [state][symbol]} probabilities
     * @return the model
     */
    public static HiddenMarkovModel discrete(double[] start, double[][] transition, double[][] emission) {
        return new HiddenMarkovModel(start.clone(), deep(transition), deep(emission), null, null);
    }

    /**
     * A model with one-dimensional Gaussian emissions.
     *
     * @param start the initial state distribution
     * @param transition {@code [from][to]} probabilities
     * @param means the emission mean per state
     * @param variances the emission variance per state
     * @return the model
     */
    public static HiddenMarkovModel gaussian(double[] start, double[][] transition, double[] means,
            double[] variances) {
        return new HiddenMarkovModel(start.clone(), deep(transition), null, means.clone(), variances.clone());
    }

    /**
     * A random discrete model to start Baum-Welch from: sticky transitions and perturbed uniform
     * emissions.
     *
     * @param states the number of states
     * @param alphabet the number of symbols
     * @param seed the seed
     * @return the model
     */
    public static HiddenMarkovModel randomDiscrete(int states, int alphabet, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] start = new double[states];
        Arrays.fill(start, 1.0 / states);
        double[][] trans = new double[states][states];
        for (int i = 0; i < states; i++) {
            for (int j = 0; j < states; j++) {
                trans[i][j] = i == j ? 0.8 : 0.2 / (states - 1);
            }
        }
        double[][] emit = new double[states][alphabet];
        for (int i = 0; i < states; i++) {
            double s = 0;
            for (int k = 0; k < alphabet; k++) {
                emit[i][k] = 1 + 0.5 * r.nextDouble();
                s += emit[i][k];
            }
            for (int k = 0; k < alphabet; k++) {
                emit[i][k] /= s;
            }
        }
        return discrete(start, trans, emit);
    }

    private static double[][] deep(double[][] a) {
        double[][] c = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i].clone();
        }
        return c;
    }

    /** @return the number of hidden states */
    public int states() {
        return start.length;
    }

    /**
     * The emission probability (or density) of an observation.
     *
     * @param state the state
     * @param observation the observation
     * @return {@code p(o | state)}
     */
    public double emission(int state, double observation) {
        if (symbols != null) {
            return symbols[state][(int) observation];
        }
        double d = observation - means[state];
        return Math.exp(-0.5 * d * d / variances[state]) / Math.sqrt(2 * Math.PI * variances[state]);
    }

    /** Scaled forward pass: alpha rows normalised, scale factors returned in {@code c}. */
    private double[][] forward(double[] obs, double[] c) {
        int n = states();
        double[][] alpha = new double[obs.length][n];
        for (int t = 0; t < obs.length; t++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                double prior;
                if (t == 0) {
                    prior = start[j];
                } else {
                    prior = 0;
                    for (int i = 0; i < n; i++) {
                        prior += alpha[t - 1][i] * transition[i][j];
                    }
                }
                alpha[t][j] = prior * Math.max(emission(j, obs[t]), 1e-300);
                sum += alpha[t][j];
            }
            c[t] = sum;
            for (int j = 0; j < n; j++) {
                alpha[t][j] /= sum;
            }
        }
        return alpha;
    }

    private double[][] backward(double[] obs, double[] c) {
        int n = states();
        double[][] beta = new double[obs.length][n];
        Arrays.fill(beta[obs.length - 1], 1);
        for (int t = obs.length - 2; t >= 0; t--) {
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int j = 0; j < n; j++) {
                    s += transition[i][j] * Math.max(emission(j, obs[t + 1]), 1e-300) * beta[t + 1][j];
                }
                beta[t][i] = s / c[t + 1];
            }
        }
        return beta;
    }

    /**
     * The log-likelihood of a sequence by the scaled forward algorithm.
     *
     * @param obs the observations
     * @return {@code log p(o_1..T)}
     */
    public double logLikelihood(double[] obs) {
        double[] c = new double[obs.length];
        forward(obs, c);
        double ll = 0;
        for (double v : c) {
            ll += Math.log(v);
        }
        return ll;
    }

    /**
     * The posterior state marginals {@code p(s_t | o_1..T)} by forward-backward.
     *
     * @param obs the observations
     * @return {@code [T][states]}
     */
    public double[][] posteriors(double[] obs) {
        double[] c = new double[obs.length];
        double[][] alpha = forward(obs, c);
        double[][] beta = backward(obs, c);
        double[][] gamma = new double[obs.length][states()];
        for (int t = 0; t < obs.length; t++) {
            for (int j = 0; j < states(); j++) {
                gamma[t][j] = alpha[t][j] * beta[t][j];
            }
        }
        return gamma;
    }

    /**
     * The most likely state path.
     *
     * @param obs the observations
     * @return the decoded states
     */
    public int[] viterbi(double[] obs) {
        int n = states();
        double[] logStart = new double[n];
        double[][] logTrans = new double[n][n];
        double[][] logEmit = new double[obs.length][n];
        for (int i = 0; i < n; i++) {
            logStart[i] = Math.log(Math.max(start[i], 1e-300));
            for (int j = 0; j < n; j++) {
                logTrans[i][j] = Math.log(Math.max(transition[i][j], 1e-300));
            }
        }
        for (int t = 0; t < obs.length; t++) {
            for (int j = 0; j < n; j++) {
                logEmit[t][j] = Math.log(Math.max(emission(j, obs[t]), 1e-300));
            }
        }
        return viterbi(logStart, logTrans, logEmit);
    }

    /**
     * Viterbi decoding of any chain with additive log-space scores: the path maximising
     * {@code start[s_0] + sum_t emit[t][s_t] + sum_t trans[s_t-1][s_t]}.
     *
     * @param logStart the start score per state
     * @param logTrans {@code [from][to]} transition scores
     * @param logEmit {@code [T][states]} emission scores
     * @return the best path
     */
    public static int[] viterbi(double[] logStart, double[][] logTrans, double[][] logEmit) {
        int steps = logEmit.length;
        int n = logStart.length;
        double[][] score = new double[steps][n];
        int[][] back = new int[steps][n];
        for (int j = 0; j < n; j++) {
            score[0][j] = logStart[j] + logEmit[0][j];
        }
        for (int t = 1; t < steps; t++) {
            for (int j = 0; j < n; j++) {
                double best = Double.NEGATIVE_INFINITY;
                int arg = 0;
                for (int i = 0; i < n; i++) {
                    double v = score[t - 1][i] + logTrans[i][j];
                    if (v > best) {
                        best = v;
                        arg = i;
                    }
                }
                score[t][j] = best + logEmit[t][j];
                back[t][j] = arg;
            }
        }
        int[] path = new int[steps];
        int last = 0;
        for (int j = 1; j < n; j++) {
            if (score[steps - 1][j] > score[steps - 1][last]) {
                last = j;
            }
        }
        path[steps - 1] = last;
        for (int t = steps - 1; t > 0; t--) {
            path[t - 1] = back[t][path[t]];
        }
        return path;
    }

    /**
     * Baum-Welch: expectation maximisation of the likelihood of unlabelled sequences, starting
     * from this model.
     *
     * @param sequences the training sequences
     * @param iterations EM iterations
     * @return the re-estimated model
     */
    public HiddenMarkovModel baumWelch(List<double[]> sequences, int iterations) {
        HiddenMarkovModel m = this;
        int n = states();
        for (int it = 0; it < iterations; it++) {
            double[] startAcc = new double[n];
            double[][] transAcc = new double[n][n];
            double[] stateAcc = new double[n];
            double[][] symAcc = symbols == null ? null : new double[n][symbols[0].length];
            double[] meanAcc = new double[n];
            double[] sqAcc = new double[n];
            for (double[] obs : sequences) {
                double[] c = new double[obs.length];
                double[][] alpha = m.forward(obs, c);
                double[][] beta = m.backward(obs, c);
                for (int t = 0; t < obs.length; t++) {
                    for (int j = 0; j < n; j++) {
                        double g = alpha[t][j] * beta[t][j];
                        if (t == 0) {
                            startAcc[j] += g;
                        }
                        stateAcc[j] += g;
                        if (symAcc != null) {
                            symAcc[j][(int) obs[t]] += g;
                        } else {
                            meanAcc[j] += g * obs[t];
                            sqAcc[j] += g * obs[t] * obs[t];
                        }
                    }
                    if (t + 1 < obs.length) {
                        for (int i = 0; i < n; i++) {
                            for (int j = 0; j < n; j++) {
                                transAcc[i][j] += alpha[t][i] * m.transition[i][j]
                                        * Math.max(m.emission(j, obs[t + 1]), 1e-300) * beta[t + 1][j] / c[t + 1];
                            }
                        }
                    }
                }
            }
            double[] ns = normalise(startAcc);
            double[][] nt = new double[n][];
            for (int i = 0; i < n; i++) {
                nt[i] = normalise(transAcc[i]);
            }
            if (symAcc != null) {
                double[][] ne = new double[n][];
                for (int i = 0; i < n; i++) {
                    for (int k = 0; k < symAcc[i].length; k++) {
                        symAcc[i][k] += 1e-6;
                    }
                    ne[i] = normalise(symAcc[i]);
                }
                m = discrete(ns, nt, ne);
            } else {
                double[] mu = new double[n];
                double[] var = new double[n];
                for (int j = 0; j < n; j++) {
                    mu[j] = meanAcc[j] / stateAcc[j];
                    var[j] = Math.max(sqAcc[j] / stateAcc[j] - mu[j] * mu[j], 1e-4);
                }
                m = gaussian(ns, nt, mu, var);
            }
        }
        return m;
    }

    private static double[] normalise(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = s > 0 ? v[i] / s : 1.0 / v.length;
        }
        return out;
    }

    /**
     * Samples a sequence and its hidden states.
     *
     * @param length the length
     * @param random the source
     * @return {observations, states as doubles}
     */
    public double[][] sample(int length, SplittableRandom random) {
        double[] obs = new double[length];
        double[] st = new double[length];
        int s = pick(start, random);
        for (int t = 0; t < length; t++) {
            if (t > 0) {
                s = pick(transition[s], random);
            }
            st[t] = s;
            obs[t] = symbols != null ? pick(symbols[s], random)
                    : means[s] + Math.sqrt(variances[s]) * random.nextGaussian();
        }
        return new double[][] {obs, st};
    }

    private static int pick(double[] p, SplittableRandom random) {
        double u = random.nextDouble();
        for (int i = 0; i < p.length - 1; i++) {
            u -= p[i];
            if (u < 0) {
                return i;
            }
        }
        return p.length - 1;
    }

    /**
     * Returns the transition probabilities.
     *
     * @return a copy of {@code [from][to]}
     */
    public double[][] transition() {
        return deep(transition);
    }

    /**
     * Returns the Gaussian emission means.
     *
     * @return a copy, or {@code null} for a discrete model
     */
    public double[] means() {
        return means == null ? null : means.clone();
    }
}
