package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.evolve.EvolutionStrategy;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Learners that account for the fact that their opponent is learning too, on an
 * {@link IteratedMatrixGame}. Two naive policy-gradient learners in the iterated prisoner's
 * dilemma drift to mutual defection; shaping the opponent's learning leads to reciprocity.
 * <ul>
 *   <li>{@link Learner#NAIVE}: {@code theta_i += lr grad_i V_i(theta_1, theta_2)}.</li>
 *   <li>{@link Learner#LOLA}: learning with opponent-learning awareness differentiates
 *       through the opponent's anticipated naive step:
 *       {@code theta_1 += lr grad_1 V_1(theta_1, theta_2 + eta grad_2 V_2(theta_1, theta_2))},
 *       with the inner gradient kept in the graph ({@code createGraph}) so that its dependence
 *       on {@code theta_1} is differentiated (Foerster et al. 2018,
 *       <a href="https://arxiv.org/abs/1709.04326">arXiv:1709.04326</a>; this is the exact
 *       look-ahead form rather than LOLA's first-order Taylor expansion).</li>
 *   <li>{@link Mfos}: model-free opponent shaping learns a meta-policy that sets its own
 *       policy from the opponent's current policy at every step of the opponent's learning,
 *       trained by an evolution strategy on the return over whole learning episodes, with no
 *       gradient through the opponent (Lu et al. 2022,
 *       <a href="https://arxiv.org/abs/2205.01447">arXiv:2205.01447</a>).</li>
 * </ul>
 */
public final class OpponentShaping {

    /** A gradient-based learner. */
    public enum Learner {
        /** Naive gradient ascent on its own value. */
        NAIVE,
        /** Learning with opponent-learning awareness. */
        LOLA
    }

    private OpponentShaping() {
    }

    /**
     * One update of one player.
     *
     * @param game the game
     * @param theta1 player 1's logits
     * @param theta2 player 2's logits
     * @param player the updating player, 0 or 1
     * @param learner its rule
     * @param learningRate its step size
     * @param lookahead the opponent step size LOLA anticipates
     * @return the player's new logits
     */
    public static double[] update(IteratedMatrixGame game, double[] theta1, double[] theta2, int player,
            Learner learner, double learningRate, double lookahead) {
        Var t1 = Var.param(IteratedMatrixGame.row(theta1));
        Var t2 = Var.param(IteratedMatrixGame.row(theta2));
        Var own = player == 0 ? t1 : t2;
        Var other = player == 0 ? t2 : t1;
        Var objective;
        if (learner == Learner.NAIVE) {
            objective = game.values(t1, t2)[player];
        } else {
            Var[] v = game.values(t1, t2);
            Var step = Var.gradients(v[1 - player], List.of(other), true).get(0);
            Var shifted = other.add(step.mul((float) lookahead));
            objective = (player == 0 ? game.values(t1, shifted) : game.values(shifted, t2))[player];
        }
        float[] g = Var.gradients(objective, List.of(own), false).get(0).value().toArray();
        double[] out = (player == 0 ? theta1 : theta2).clone();
        for (int i = 0; i < out.length; i++) {
            out[i] += learningRate * g[i];
        }
        return out;
    }

    /**
     * Trains two learners against each other with simultaneous updates from random logits.
     *
     * @param game the game
     * @param first player 1's rule
     * @param second player 2's rule
     * @param learningRate both step sizes
     * @param lookahead LOLA's anticipated opponent step size
     * @param steps updates
     * @param seed the seed of the initial logits (standard normal)
     * @return {@code {theta1, theta2}} after training
     */
    public static double[][] train(IteratedMatrixGame game, Learner first, Learner second, double learningRate,
            double lookahead, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] t1 = gaussian(r);
        double[] t2 = gaussian(r);
        for (int s = 0; s < steps; s++) {
            double[] n1 = update(game, t1, t2, 0, first, learningRate, lookahead);
            double[] n2 = update(game, t1, t2, 1, second, learningRate, lookahead);
            t1 = n1;
            t2 = n2;
        }
        return new double[][] {t1, t2};
    }

    private static double[] gaussian(SplittableRandom r) {
        double[] t = new double[5];
        for (int i = 0; i < 5; i++) {
            t[i] = r.nextGaussian();
        }
        return t;
    }

    /**
     * Model-free opponent shaping against naive learners: a linear meta-policy maps the
     * opponent's five cooperation probabilities (and a bias) to the shaper's five logits.
     * One meta-episode lets a fresh naive opponent (random initial logits) learn for
     * {@code horizon} steps against the meta-policy; its fitness is the shaper's mean value
     * over the episode. The meta-policy is trained by {@link EvolutionStrategy}.
     */
    public static final class Mfos {
        private final IteratedMatrixGame game;
        private final int horizon;
        private final double opponentRate;
        private double[] weights = new double[30];

        private Mfos(IteratedMatrixGame game, int horizon, double opponentRate) {
            this.game = game;
            this.horizon = horizon;
            this.opponentRate = opponentRate;
        }

        /**
         * @param game the game
         * @param horizon opponent learning steps per meta-episode
         * @param opponentRate the naive opponent's step size
         * @return an untrained shaper
         */
        public static Mfos of(IteratedMatrixGame game, int horizon, double opponentRate) {
            return new Mfos(Objects.requireNonNull(game), horizon, opponentRate);
        }

        private double[] policy(double[] w, double[] opponent) {
            double[] q = IteratedMatrixGame.cooperation(opponent);
            double[] out = new double[5];
            for (int i = 0; i < 5; i++) {
                double z = w[25 + i];
                for (int j = 0; j < 5; j++) {
                    z += w[5 * i + j] * q[j];
                }
                out[i] = z;
            }
            return out;
        }

        /**
         * Plays one meta-episode against a fresh naive learner.
         *
         * @param w meta-policy weights
         * @param seed the opponent's initial logits
         * @return {@code {shaper's mean value, opponent's mean value}}
         */
        private double[] episode(double[] w, long seed) {
            double[] opp = gaussian(new SplittableRandom(seed));
            double[] total = new double[2];
            for (int t = 0; t < horizon; t++) {
                double[] own = policy(w, opp);
                double[] v = game.values(own, opp);
                total[0] += v[0] / horizon;
                total[1] += v[1] / horizon;
                double[] g = game.gradient(own, opp, 1, 1);
                for (int i = 0; i < 5; i++) {
                    opp[i] += opponentRate * g[i];
                }
            }
            return total;
        }

        /**
         * Trains the meta-policy.
         *
         * @param iterations evolution-strategy iterations
         * @param opponentsPerFitness meta-episodes averaged per fitness evaluation
         * @param seed the seed
         * @return this
         */
        public Mfos train(int iterations, int opponentsPerFitness, long seed) {
            EvolutionStrategy es = EvolutionStrategy.of(weights).sigma(0.3).learningRate(0.1).pairs(10).seed(seed);
            int[] round = {0};
            es.run(w -> {
                double total = 0;
                for (int k = 0; k < opponentsPerFitness; k++) {
                    total += episode(w, seed * 1000 + round[0] % 50 * 7 + k)[0];
                }
                round[0]++;
                return total / opponentsPerFitness;
            }, iterations);
            weights = es.center();
            return this;
        }

        /**
         * Evaluates the shaper against fresh naive learners.
         *
         * @param opponents meta-episodes
         * @param seed their seeds' base
         * @return {@code {shaper's mean value, opponent's mean value}}
         */
        public double[] evaluate(int opponents, long seed) {
            double[] total = new double[2];
            for (int k = 0; k < opponents; k++) {
                double[] r = episode(weights, seed + 7919L * k);
                total[0] += r[0] / opponents;
                total[1] += r[1] / opponents;
            }
            return total;
        }
    }
}
