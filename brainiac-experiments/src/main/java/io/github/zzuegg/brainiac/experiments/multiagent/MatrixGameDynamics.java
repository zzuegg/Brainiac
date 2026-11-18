package io.github.zzuegg.brainiac.experiments.multiagent;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.Exploitability;
import io.github.zzuegg.brainiac.experimental.rl.GameTree;
import io.github.zzuegg.brainiac.experimental.rl.IteratedMatrixGame;
import io.github.zzuegg.brainiac.experimental.rl.MatrixGame;
import io.github.zzuegg.brainiac.experimental.rl.OpponentShaping;
import io.github.zzuegg.brainiac.experimental.rl.RegularizedEquilibriumLearner;
import io.github.zzuegg.brainiac.experimental.train.MinMaxOptimizers;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Learning dynamics in small games, where the equilibrium is known exactly.
 * <ul>
 *   <li>{@link #minMax}: distance to the equilibrium after a fixed number of steps on the
 *       bilinear game {@code min_x max_y x.y} and the Dirac-GAN (Mescheder et al. 2018), for
 *       simultaneous and alternating gradient descent-ascent (plain and with Adam) against
 *       extragradient, optimistic GDA, optimistic Adam and consensus optimisation
 *       ({@link MinMaxOptimizers}, W7 5.5).</li>
 *   <li>{@link #lastIterate}: exploitability of the last iterate on matching pennies and
 *       biased rock-paper-scissors started off-equilibrium, for multiplicative weights
 *       against magnetic mirror descent, R-NaD and Nash policy gradient (W7 5.4).</li>
 *   <li>{@link #shaping}: average reward per step in the iterated prisoner's dilemma for
 *       naive learners, LOLA and M-FOS ({@link OpponentShaping}, W7 5.8).</li>
 * </ul>
 */
public final class MatrixGameDynamics {

    /** A two-player differentiable game with its equilibrium at the origin. */
    public enum ContinuousGame {
        /** {@code f(x, y) = x . y} with 4-dimensional {@code x} and {@code y}. */
        BILINEAR,
        /** The Dirac-GAN: {@code f(theta, psi) = phi(psi theta) + phi(0)}, {@code phi(t) = -log(1 + e^-t)}. */
        DIRAC_GAN
    }

    private MatrixGameDynamics() {
    }

    /** The optimisers compared by {@link #minMax}. */
    public static final List<String> METHODS = List.of("simultaneous GDA", "alternating GDA", "simultaneous Adam",
            "alternating Adam", "extragradient", "optimistic GDA", "optimistic Adam", "consensus");

    /**
     * @param game the game
     * @param steps optimisation steps
     * @param learningRate the step size of every method
     * @return method to final distance from the equilibrium (the start is at distance 1 per
     *     coordinate pair, about 2.8 for the bilinear game and 1.4 for the Dirac-GAN)
     */
    public static Map<String, Double> minMax(ContinuousGame game, int steps, float learningRate) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (String method : METHODS) {
            int dim = game == ContinuousGame.BILINEAR ? 4 : 1;
            float[] ones = new float[dim];
            java.util.Arrays.fill(ones, 1f);
            Var x = Var.param(Tensor.of(Shape.of(1, dim), ones.clone()));
            Var y = Var.param(Tensor.of(Shape.of(1, dim), ones.clone()));
            MinMaxOptimizers.Game g = () -> {
                Var f = game == ContinuousGame.BILINEAR ? x.mul(y).sum()
                        : x.mul(y).neg().softplus().neg().sum().sub((float) Math.log(2));
                return new Var[] {f, f.neg()};
            };
            List<Var> px = List.of(x);
            List<Var> py = List.of(y);
            MinMaxOptimizers.Stepper stepper = switch (method) {
                case "simultaneous GDA" -> MinMaxOptimizers.simultaneous(px, Optimizer.sgd(learningRate), py,
                        Optimizer.sgd(learningRate));
                case "alternating GDA" -> MinMaxOptimizers.alternating(px, Optimizer.sgd(learningRate), py,
                        Optimizer.sgd(learningRate));
                case "simultaneous Adam" -> MinMaxOptimizers.simultaneous(px, Optimizer.adam(learningRate), py,
                        Optimizer.adam(learningRate));
                case "alternating Adam" -> MinMaxOptimizers.alternating(px, Optimizer.adam(learningRate), py,
                        Optimizer.adam(learningRate));
                case "extragradient" -> MinMaxOptimizers.extragradient(px, Optimizer.sgd(learningRate), py,
                        Optimizer.sgd(learningRate), learningRate);
                case "optimistic GDA" -> MinMaxOptimizers.simultaneous(px, MinMaxOptimizers.optimisticSgd(learningRate),
                        py, MinMaxOptimizers.optimisticSgd(learningRate));
                case "optimistic Adam" -> MinMaxOptimizers.simultaneous(px,
                        MinMaxOptimizers.optimisticAdam(learningRate, 0.5f, 0.9f), py,
                        MinMaxOptimizers.optimisticAdam(learningRate, 0.5f, 0.9f));
                case "consensus" -> MinMaxOptimizers.consensus(px, Optimizer.sgd(learningRate), py,
                        Optimizer.sgd(learningRate), 1f);
                default -> throw new IllegalStateException(method);
            };
            for (int s = 0; s < steps; s++) {
                stepper.step(g);
            }
            double d = 0;
            for (float v : x.value().toArray()) {
                d += v * v;
            }
            for (float v : y.value().toArray()) {
                d += v * v;
            }
            out.put(method, Math.sqrt(d));
        }
        return out;
    }

    /**
     * Last-iterate exploitability from an off-equilibrium start (player 0 plays the first
     * action with probability 0.9).
     *
     * @param game the matrix game
     * @param iterations simultaneous updates
     * @return method to the exploitability of the final policies
     */
    public static Map<String, Double> lastIterate(MatrixGame game, int iterations) {
        GameTree tree = GameTree.compile(game);
        double[][] start = tree.uniformPolicy();
        int rows = game.rows();
        for (int a = 0; a < rows; a++) {
            start[0][a] = a == 0 ? 0.9 : 0.1 / (rows - 1);
        }
        Map<String, Double> out = new LinkedHashMap<>();
        RegularizedEquilibriumLearner[] learners = {
            RegularizedEquilibriumLearner.unregularized(tree, 0.1),
            RegularizedEquilibriumLearner.mmd(tree, 0.5, 0.1),
            RegularizedEquilibriumLearner.rnad(tree, 0.5, 0.1, 100),
            RegularizedEquilibriumLearner.nashPg(tree, 0.5, 0.5, 100)};
        String[] names = {"multiplicative weights", "MMD", "R-NaD", "Nash policy gradient"};
        for (int i = 0; i < learners.length; i++) {
            learners[i].start(start).run(iterations);
            out.put(names[i], Exploitability.of(tree, learners[i].policy()));
        }
        return out;
    }

    /**
     * The outcome of a pairing in the iterated prisoner's dilemma.
     *
     * @param first player 1's mean value per step (mutual cooperation -1, mutual defection -2)
     * @param second player 2's mean value per step
     * @param cooperateAfterCooperation player 1's probability of cooperating after mutual
     *     cooperation
     * @param cooperateAfterDefection player 1's probability of cooperating after it cooperated
     *     and the opponent defected
     */
    public record Pairing(double first, double second, double cooperateAfterCooperation,
                          double cooperateAfterDefection) {
    }

    /**
     * Naive learners, LOLA and M-FOS in the iterated prisoner's dilemma (gamma 0.96),
     * averaged over seeds.
     *
     * @param steps learning steps of the gradient learners
     * @param seeds seeds averaged
     * @param mfosIterations evolution-strategy iterations of the M-FOS meta-policy
     * @return "naive vs naive", "LOLA vs LOLA", "LOLA vs naive" and "M-FOS vs naive"
     */
    public static Map<String, Pairing> shaping(int steps, int seeds, int mfosIterations) {
        IteratedMatrixGame ipd = IteratedMatrixGame.prisonersDilemma(0.96);
        Map<String, Pairing> out = new LinkedHashMap<>();
        OpponentShaping.Learner naive = OpponentShaping.Learner.NAIVE;
        OpponentShaping.Learner lola = OpponentShaping.Learner.LOLA;
        OpponentShaping.Learner[][] pairs = {{naive, naive}, {lola, lola}, {lola, naive}};
        String[] names = {"naive vs naive", "LOLA vs LOLA", "LOLA vs naive"};
        for (int k = 0; k < pairs.length; k++) {
            double[] sum = new double[4];
            for (long seed = 1; seed <= seeds; seed++) {
                double[][] t = OpponentShaping.train(ipd, pairs[k][0], pairs[k][1], 10, 10, steps, seed);
                double[] v = ipd.values(t[0], t[1]);
                double[] c = IteratedMatrixGame.cooperation(t[0]);
                sum[0] += v[0] / seeds;
                sum[1] += v[1] / seeds;
                sum[2] += c[1] / seeds;
                sum[3] += c[2] / seeds;
            }
            out.put(names[k], new Pairing(sum[0], sum[1], sum[2], sum[3]));
        }
        double[] m = new double[2];
        for (long seed = 1; seed <= seeds; seed++) {
            double[] r = OpponentShaping.Mfos.of(ipd, 30, 1).train(mfosIterations, 2, seed).evaluate(10, 1000 + seed);
            m[0] += r[0] / seeds;
            m[1] += r[1] / seeds;
        }
        out.put("M-FOS vs naive", new Pairing(m[0], m[1], Double.NaN, Double.NaN));
        double[] base = new double[2];
        for (long seed = 1; seed <= seeds; seed++) {
            double[] r = OpponentShaping.Mfos.of(ipd, 30, 1).evaluate(10, 1000 + seed);
            base[0] += r[0] / seeds;
            base[1] += r[1] / seeds;
        }
        out.put("untrained meta-policy vs naive", new Pairing(base[0], base[1], Double.NaN, Double.NaN));
        return out;
    }
}
