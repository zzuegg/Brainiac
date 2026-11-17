package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.experimental.nn.PBitIsingMachine;

import java.util.SplittableRandom;

/**
 * Probabilistic computing with p-bits (Camsari et al., <a href="https://arxiv.org/abs/1610.00377">1610.00377</a>;
 * Aadit et al., "Massively parallel probabilistic computing with sparse Ising machines",
 * <a href="https://arxiv.org/abs/2110.02481">2110.02481</a>): an invertible AND gate whose
 * Ising energy is lowest exactly on the truth table, run forwards (clamp the inputs) and
 * backwards (clamp the output and sample the inputs), and the ferromagnetic phase
 * transition of a 2-D Ising lattice (Onsager: {@code beta_c = ln(1 + sqrt 2) / 2 ~ 0.4407}).
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
}
