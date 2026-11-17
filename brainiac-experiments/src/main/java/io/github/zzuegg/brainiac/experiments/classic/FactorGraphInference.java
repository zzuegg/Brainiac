package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.experimental.ml.FactorGraph;
import io.github.zzuegg.brainiac.experimental.nn.PBitIsingMachine;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Exact and approximate inference in discrete graphical models: variable elimination
 * (exact, the reference), loopy belief propagation and Gibbs sampling ({@link FactorGraph}),
 * plus the p-bit network ({@link PBitIsingMachine}) as a hardware-style Gibbs sampler on the
 * Ising grid. Three models: the classic sprinkler network (a loop: cloudy, sprinkler, rain, wet
 * grass) with the grass observed wet; the alarm network (a tree) with both neighbours calling;
 * and a 4x4 spin glass with random couplings. Reported: the mean absolute marginal error
 * against variable elimination, and the time taken.
 */
public final class FactorGraphInference {

    private FactorGraphInference() {
    }

    /**
     * One method's score.
     *
     * @param error the mean absolute marginal error against exact inference
     * @param millis the time taken in milliseconds
     */
    public record Score(double error, double millis) {
    }

    /** @return the sprinkler network with wet grass observed (variables C, S, R, W) */
    public static FactorGraph sprinkler() {
        FactorGraph g = new FactorGraph();
        int c = g.variable(2);
        int s = g.variable(2);
        int r = g.variable(2);
        int w = g.variable(2);
        g.factor(new int[] {c}, new double[] {0.5, 0.5});
        g.conditional(s, new int[] {c}, new double[] {0.5, 0.5, 0.9, 0.1});
        g.conditional(r, new int[] {c}, new double[] {0.8, 0.2, 0.2, 0.8});
        g.conditional(w, new int[] {s, r}, new double[] {1, 0, 0.1, 0.9, 0.1, 0.9, 0.01, 0.99});
        g.observe(w, 1);
        return g;
    }

    /** @return the alarm network with John and Mary calling (variables B, E, A, J, M) */
    public static FactorGraph alarm() {
        FactorGraph g = new FactorGraph();
        int b = g.variable(2);
        int e = g.variable(2);
        int a = g.variable(2);
        int j = g.variable(2);
        int m = g.variable(2);
        g.factor(new int[] {b}, new double[] {0.99, 0.01});
        g.factor(new int[] {e}, new double[] {0.98, 0.02});
        g.conditional(a, new int[] {b, e}, new double[] {0.999, 0.001, 0.71, 0.29, 0.06, 0.94, 0.05, 0.95});
        g.conditional(j, new int[] {a}, new double[] {0.95, 0.05, 0.1, 0.9});
        g.conditional(m, new int[] {a}, new double[] {0.99, 0.01, 0.3, 0.7});
        g.observe(j, 1);
        g.observe(m, 1);
        return g;
    }

    /**
     * Random spin-glass couplings on an open square grid.
     *
     * @param side the grid side
     * @param seed the seed
     * @return {couplings, fields} as {@code double[2][][]}; fields in row 0 of the second entry
     */
    public static double[][][] spinGlass(int side, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = side * side;
        double[][] j = new double[n][n];
        double[] h = new double[n];
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                int i = y * side + x;
                h[i] = r.nextDouble(-0.3, 0.3);
                if (x + 1 < side) {
                    j[i][i + 1] = j[i + 1][i] = r.nextDouble(-0.6, 0.6);
                }
                if (y + 1 < side) {
                    j[i][i + side] = j[i + side][i] = r.nextDouble(-0.6, 0.6);
                }
            }
        }
        return new double[][][] {j, {h}};
    }

    /**
     * Runs every method on one graph.
     *
     * @param g the graph
     * @param gibbsSweeps Gibbs sweeps
     * @param seed the seed
     * @return method to score
     */
    public static Map<String, Score> compare(FactorGraph g, int gibbsSweeps, long seed) {
        Map<String, Score> out = new LinkedHashMap<>();
        long t0 = System.nanoTime();
        double[][] exact = g.exactMarginals();
        out.put("variable elimination", new Score(0, ms(t0)));
        t0 = System.nanoTime();
        double[][] bp = g.loopyBeliefPropagation(200, 0.3);
        out.put("loopy BP", new Score(FactorGraph.marginalError(bp, exact), ms(t0)));
        t0 = System.nanoTime();
        double[][] gibbs = g.gibbs(gibbsSweeps, 200, seed);
        out.put("Gibbs", new Score(FactorGraph.marginalError(gibbs, exact), ms(t0)));
        return out;
    }

    /**
     * The spin glass, including the p-bit sampler.
     *
     * @param side the grid side
     * @param sweeps sampling sweeps for Gibbs and the p-bits
     * @param seed the seed
     * @return method to score
     */
    public static Map<String, Score> isingGrid(int side, int sweeps, long seed) {
        double[][][] model = spinGlass(side, seed);
        double[][] j = model[0];
        double[] h = model[1][0];
        FactorGraph g = FactorGraph.ising(j, h, 1);
        Map<String, Score> out = compare(g, sweeps, seed);
        double[][] exact = g.exactMarginals();
        long t0 = System.nanoTime();
        PBitIsingMachine machine = PBitIsingMachine.of(j, h);
        SplittableRandom r = new SplittableRandom(seed);
        machine.randomize(r);
        int n = h.length;
        double[] up = new double[n];
        for (int s = 0; s < sweeps + 200; s++) {
            machine.sweep(1, r);
            if (s >= 200) {
                int[] spins = machine.spins();
                for (int i = 0; i < n; i++) {
                    up[i] += spins[i] > 0 ? 1 : 0;
                }
            }
        }
        double[][] pbit = new double[n][2];
        for (int i = 0; i < n; i++) {
            pbit[i][1] = up[i] / sweeps;
            pbit[i][0] = 1 - pbit[i][1];
        }
        out.put("p-bit machine", new Score(FactorGraph.marginalError(pbit, exact), ms(t0)));
        return out;
    }

    private static double ms(long t0) {
        return (System.nanoTime() - t0) / 1e6;
    }
}
