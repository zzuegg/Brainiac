package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.experimental.ml.SimulatedBifurcation;
import io.github.zzuegg.brainiac.experimental.ml.SpectralClustering;
import io.github.zzuegg.brainiac.experimental.nn.PBitIsingMachine;

import java.util.SplittableRandom;

/**
 * Probabilistic computing with p-bits (Camsari et al., <a href="https://arxiv.org/abs/1610.00377">1610.00377</a>;
 * Aadit et al., "Massively parallel probabilistic computing with sparse Ising machines",
 * <a href="https://arxiv.org/abs/2110.02481">2110.02481</a>): an invertible AND gate whose
 * Ising energy is lowest exactly on the truth table, run forwards (clamp the inputs) and
 * backwards (clamp the output and sample the inputs), and the ferromagnetic phase
 * transition of a 2-D Ising lattice (Onsager: {@code beta_c = ln(1 + sqrt 2) / 2 ~ 0.4407}).
 *
 * <p>Wave 8 (8.6.18) adds Ising-solver rows ({@link #ising}): MaxCut on random and planted
 * graphs and Sherrington–Kirkpatrick instances, solved by {@link SimulatedBifurcation}
 * (adiabatic, ballistic, discrete), p-bit annealing, simulated annealing and the spectral cut.
 */
public final class ProbabilisticBits {
    private ProbabilisticBits() {
    }

    /**
     * The invertible AND gate on p-bits (A, B, C = A and B), bipolar (+1 true, -1 false):
     * every truth-table row has energy -3, every other state at least -1.
     *
     * @return the machine
     */
    public static PBitIsingMachine andGate() {
        double[][] j = {{0, -1, 2}, {-1, 0, 2}, {2, 2, 0}};
        return PBitIsingMachine.of(j, new double[] {1, 1, -2});
    }

    /**
     * Samples the state distribution by Gibbs sweeps.
     *
     * @param machine the machine, at most 20 bits
     * @param beta the inverse temperature
     * @param sweeps sampled sweeps
     * @param burnIn sweeps discarded first
     * @param seed the seed
     * @return the visit share of each state code
     */
    public static double[] histogram(PBitIsingMachine machine, double beta, int sweeps, int burnIn, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        machine.randomize(r);
        double[] counts = new double[1 << machine.size()];
        for (int s = 0; s < burnIn + sweeps; s++) {
            machine.sweep(beta, r);
            if (s >= burnIn) {
                counts[machine.code()] += 1.0 / sweeps;
            }
        }
        return counts;
    }

    /**
     * The total variation distance between two distributions.
     *
     * @param p the first
     * @param q the second
     * @return {@code 1/2 sum |p - q|}
     */
    public static double totalVariation(double[] p, double[] q) {
        double d = 0;
        for (int i = 0; i < p.length; i++) {
            d += Math.abs(p[i] - q[i]);
        }
        return d / 2;
    }

    /**
     * Mean absolute magnetization of a periodic square lattice, started cold (all +1).
     *
     * @param side the lattice side
     * @param beta the inverse temperature
     * @param sweeps sampled sweeps
     * @param burnIn discarded sweeps
     * @param seed the seed
     * @return {@code <|m|>}
     */
    public static double latticeMagnetization(int side, double beta, int sweeps, int burnIn, long seed) {
        PBitIsingMachine m = PBitIsingMachine.lattice(side);
        SplittableRandom r = new SplittableRandom(seed);
        double sum = 0;
        for (int s = 0; s < burnIn + sweeps; s++) {
            m.sweep(beta, r);
            if (s >= burnIn) {
                sum += Math.abs(m.magnetization());
            }
        }
        return sum / sweeps;
    }

    // ---------------------------------------------------------------------------------------
    // Wave 8 item 8.6.18: Ising rows. MaxCut and Sherrington-Kirkpatrick instances solved by
    // simulated bifurcation (SimulatedBifurcation), p-bit annealing (PBitIsingMachine),
    // simulated annealing and, for MaxCut, the spectral-clustering cut.
    // ---------------------------------------------------------------------------------------

    /** An Ising solver. */
    public enum IsingSolver {
        /** adiabatic simulated bifurcation */
        SB_ADIABATIC,
        /** ballistic simulated bifurcation */
        SB_BALLISTIC,
        /** discrete simulated bifurcation */
        SB_DISCRETE,
        /** p-bit Gibbs annealing */
        PBIT_ANNEALING,
        /** single-spin Metropolis annealing */
        SIMULATED_ANNEALING,
        /** two-way spectral clustering of the graph (MaxCut only) */
        SPECTRAL
    }

    /**
     * A solver's results over several runs.
     *
     * @param best the best objective (cut value, or energy per spin for SK)
     * @param mean the mean objective
     * @param millis the mean wall time per run
     */
    public record IsingResult(double best, double mean, double millis) {
    }

    /**
     * A MaxCut instance: an Erdős–Rényi graph ({@code planted = false}) or a planted
     * two-block partition with few edges inside the blocks and many across
     * ({@code SyntheticGraphs.communities}).
     *
     * @param nodes the node count (even)
     * @param density the mean edge probability
     * @param planted plant a cut
     * @param seed the seed
     * @return the 0/1 weights
     */
    public static double[][] maxCutGraph(int nodes, double density, boolean planted, long seed) {
        SyntheticGraphs.Graph g = planted
                ? SyntheticGraphs.communities(2, nodes / 2, density * 0.4, density * 1.6, 0, seed)
                : SyntheticGraphs.communities(1, nodes, density, density, 0, seed);
        double[][] w = new double[nodes][nodes];
        for (int i = 0; i < nodes; i++) {
            for (int j = 0; j < nodes; j++) {
                w[i][j] = g.adjacency()[i * nodes + j];
            }
        }
        return w;
    }

    /**
     * A Sherrington–Kirkpatrick instance, {@code J_ij = ±1/√N}; its ground-state energy per
     * spin tends to about -0.763 (Parisi).
     *
     * @param n spins
     * @param seed the seed
     * @return the couplings
     */
    public static double[][] sherringtonKirkpatrick(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] j = new double[n][n];
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                j[a][b] = j[b][a] = (r.nextBoolean() ? 1 : -1) / Math.sqrt(n);
            }
        }
        return j;
    }

    /**
     * The cut value of a partition.
     *
     * @param w the weights
     * @param s spins
     * @return the total weight of edges across
     */
    public static double cut(double[][] w, int[] s) {
        double c = 0;
        for (int i = 0; i < w.length; i++) {
            for (int j = i + 1; j < w.length; j++) {
                c += s[i] != s[j] ? w[i][j] : 0;
            }
        }
        return c;
    }

    /**
     * Solves MaxCut ({@code maxCut = true}, couplings {@code J = -w}, objective the cut) or an
     * Ising instance with couplings {@code w} (objective the energy per spin).
     *
     * @param w the weights or couplings
     * @param maxCut whether {@code w} is a MaxCut graph
     * @param solver the solver
     * @param runs independent runs
     * @param effort integration steps or sweeps per run
     * @param seed the seed
     * @return best and mean objective and time per run
     */
    public static IsingResult ising(double[][] w, boolean maxCut, IsingSolver solver, int runs, int effort,
            long seed) {
        int n = w.length;
        double[][] j = new double[n][n];
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) {
                j[a][b] = maxCut ? -w[a][b] : w[a][b];
            }
        }
        double[] h = new double[n];
        SimulatedBifurcation sb = SimulatedBifurcation.of(j, h);
        PBitIsingMachine pbits = solver == IsingSolver.PBIT_ANNEALING ? PBitIsingMachine.of(j, h) : null;
        double best = maxCut ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
        double sum = 0;
        long nanos = 0;
        for (int run = 0; run < runs; run++) {
            long start = System.nanoTime();
            long s = seed + 1000L * run;
            int[] spins = switch (solver) {
                case SB_ADIABATIC -> sb.solve(SimulatedBifurcation.Variant.ADIABATIC, effort, 0.5, s).spins();
                case SB_BALLISTIC -> sb.solve(SimulatedBifurcation.Variant.BALLISTIC, effort, 1.0, s).spins();
                case SB_DISCRETE -> sb.solve(SimulatedBifurcation.Variant.DISCRETE, effort, 1.0, s).spins();
                case SIMULATED_ANNEALING -> sb.anneal(effort, 0.1 * scaleOf(j), 3 * scaleOf(j), s).spins();
                case PBIT_ANNEALING -> pbitAnneal(pbits, effort, 0.1 * scaleOf(j), 3 * scaleOf(j), s);
                case SPECTRAL -> {
                    if (!maxCut) {
                        throw new IllegalArgumentException("the spectral cut needs a MaxCut graph");
                    }
                    int[] c = SpectralClustering.cluster(w, 2, s);
                    int[] out = new int[n];
                    for (int i = 0; i < n; i++) {
                        out[i] = c[i] == 0 ? 1 : -1;
                    }
                    yield out;
                }
            };
            nanos += System.nanoTime() - start;
            double objective = maxCut ? cut(w, spins) : sb.energy(spins) / n;
            sum += objective;
            best = maxCut ? Math.max(best, objective) : Math.min(best, objective);
        }
        return new IsingResult(best, sum / runs, nanos / 1e6 / runs);
    }

    /** {@code 1 / RMS coupling of a spin's local field}: the inverse-temperature unit. */
    private static double scaleOf(double[][] j) {
        double s = 0;
        for (double[] row : j) {
            for (double v : row) {
                s += v * v;
            }
        }
        return 1 / Math.sqrt(Math.max(1e-12, s / j.length));
    }

    private static int[] pbitAnneal(PBitIsingMachine m, int sweeps, double betaStart, double betaEnd, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        m.randomize(r);
        int[] best = m.spins();
        double bestEnergy = m.energy(best);
        for (int s = 0; s < sweeps; s++) {
            double beta = betaStart * Math.pow(betaEnd / betaStart, s / (double) Math.max(1, sweeps - 1));
            m.sweep(beta, r);
            if (s % 10 == 9 || s == sweeps - 1) {
                int[] now = m.spins();
                double e = m.energy(now);
                if (e < bestEnergy) {
                    bestEnergy = e;
                    best = now;
                }
            }
        }
        return best;
    }
}
