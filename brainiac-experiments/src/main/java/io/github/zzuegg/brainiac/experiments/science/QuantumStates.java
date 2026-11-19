package io.github.zzuegg.brainiac.experiments.science;

import io.github.zzuegg.brainiac.experimental.quantum.ConditionalQuantumState;
import io.github.zzuegg.brainiac.experimental.quantum.NeuralQuantumState;
import io.github.zzuegg.brainiac.experimental.quantum.PauliHamiltonian;
import io.github.zzuegg.brainiac.experimental.quantum.SpinModels;
import io.github.zzuegg.brainiac.experimental.quantum.StateVector;
import io.github.zzuegg.brainiac.experimental.quantum.Tomography;
import io.github.zzuegg.brainiac.experimental.quantum.VariationalMonteCarlo;
import io.github.zzuegg.brainiac.experimental.quantum.Vqe;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.DoubleFunction;

/**
 * Neural quantum states, foundation quantum states and tomography (wave 13, items 13.3.3 to
 * 13.3.5).
 *
 * <ul>
 *   <li>{@link #nqs}: variational Monte Carlo of RBM, Jastrow and MLP states on the TFIM and of
 *       an RBM with Marshall's sign on the Heisenberg chain; stochastic reconfiguration against
 *       Adam, exact sums against Metropolis sampling, and the mean-field and VQE baselines.</li>
 *   <li>{@link #foundation}: one coupling-conditioned RBM for the TFIM over
 *       {@code h in [0.2, 2]} against independent per-coupling RBMs, at a quarter of their
 *       total cost; energy errors on a grid of (mostly unseen) couplings and the location of
 *       the fidelity-susceptibility peak against the exact one.</li>
 *   <li>{@link #tomography}: infidelity of linear inversion and maximum likelihood against
 *       shots, neural RBM tomography of a GHZ state, and classical shadows against direct
 *       per-observable measurement for all 1- and 2-local Paulis.</li>
 * </ul>
 */
public final class QuantumStates {

    private QuantumStates() {
    }

    private static double rel(double e, double e0) {
        return Math.abs((e - e0) / e0);
    }

    /**
     * Neural-quantum-state table.
     *
     * @param n spins
     * @param steps optimisation steps
     * @param seed the seed
     * @return metric to values
     */
    public static Map<String, double[]> nqs(int n, int steps, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        PauliHamiltonian h = SpinModels.tfim(n, 1, 1, true);
        double e0 = h.groundState(100, seed).energy();
        VariationalMonteCarlo.Result sr = VariationalMonteCarlo.run(h, NeuralQuantumState.rbm(n, 2, seed),
                VariationalMonteCarlo.Method.STOCHASTIC_RECONFIGURATION, steps, 0.05, 1e-3, 0, false, seed);
        VariationalMonteCarlo.Result adam = VariationalMonteCarlo.run(h, NeuralQuantumState.rbm(n, 2, seed),
                VariationalMonteCarlo.Method.ADAM, steps, 0.01, 0, 0, false, seed);
        VariationalMonteCarlo.Result srSampled = VariationalMonteCarlo.run(h, NeuralQuantumState.rbm(n, 2, seed),
                VariationalMonteCarlo.Method.STOCHASTIC_RECONFIGURATION, steps / 2, 0.05, 1e-3, 400, false, seed);
        VariationalMonteCarlo.Result jas = VariationalMonteCarlo.run(h, NeuralQuantumState.jastrow(n, seed),
                VariationalMonteCarlo.Method.STOCHASTIC_RECONFIGURATION, steps, 0.05, 1e-3, 0, false, seed);
        VariationalMonteCarlo.Result mlp = VariationalMonteCarlo.run(h, NeuralQuantumState.mlp(n, 2 * n, seed),
                VariationalMonteCarlo.Method.STOCHASTIC_RECONFIGURATION, steps, 0.05, 1e-3, 0, false, seed);
        // energy variance at initialisation, for the variance-reduction ratio
        double var0 = VariationalMonteCarlo.estimate(h, NeuralQuantumState.rbm(n, 2, seed), null, false).variance();
        out.put("TFIM h=1 rel err {RBM SR, RBM Adam, RBM SR sampled, Jastrow SR, MLP SR}", new double[] {
            rel(sr.energy(), e0), rel(adam.energy(), e0), rel(srSampled.energy(), e0), rel(jas.energy(), e0),
            rel(mlp.energy(), e0)});
        out.put("E_loc variance {init, SR, Adam}", new double[] {var0, sr.variance(), adam.variance()});
        out.put("sampled final step estimate vs exact {estimate, exact}", new double[] {
            srSampled.trace()[srSampled.trace().length - 1], srSampled.energy()});
        // steps for SR and Adam to reach 1e-2 relative error (from the traces)
        out.put("steps to 1e-2 {SR, Adam}", new double[] {firstBelow(sr.trace(), e0, 1e-2),
            firstBelow(adam.trace(), e0, 1e-2)});
        // baselines
        Vqe mf = new Vqe(Vqe.product(n), h);
        Vqe hea = new Vqe(Vqe.hardwareEfficient(n, 2), h);
        out.put("baselines rel err {mean field, VQE HEA depth 2}", new double[] {
            rel(mf.bestOf(2, 150, 1.0, seed).energy(), e0), rel(hea.bestOf(2, 150, 0.1, seed).energy(), e0)});
        out.put("params {RBM a=2, Jastrow, MLP, HEA}", new double[] {NeuralQuantumState.rbm(n, 2, 0).parameterCount(),
            NeuralQuantumState.jastrow(n, 0).parameterCount(), NeuralQuantumState.mlp(n, 2 * n, 0).parameterCount(),
            hea.circuit().parameterCount()});
        // Heisenberg chain with and without Marshall's sign
        PauliHamiltonian xxz = SpinModels.xxz(n, 1, 1, true);
        double ex = xxz.groundState(100, seed).energy();
        VariationalMonteCarlo.Result withSign = VariationalMonteCarlo.run(xxz,
                NeuralQuantumState.rbm(n, 2, seed).marshallSign(true),
                VariationalMonteCarlo.Method.STOCHASTIC_RECONFIGURATION, steps, 0.05, 1e-3, 0, false, seed);
        VariationalMonteCarlo.Result noSign = VariationalMonteCarlo.run(xxz, NeuralQuantumState.rbm(n, 2, seed),
                VariationalMonteCarlo.Method.STOCHASTIC_RECONFIGURATION, steps, 0.05, 1e-3, 0, false, seed);
        out.put("Heisenberg rel err {RBM+Marshall SR, RBM no sign SR}", new double[] {rel(withSign.energy(), ex),
            rel(noSign.energy(), ex)});
        return out;
    }

    private static double firstBelow(double[] trace, double e0, double tol) {
        for (int i = 0; i < trace.length; i++) {
            if (rel(trace[i], e0) < tol) {
                return i + 1;
            }
        }
        return Double.NaN;
    }

    /**
     * Foundation-NQS table.
     *
     * @param n spins
     * @param grid evaluation couplings
     * @param perHSteps SR steps of each independent per-coupling RBM
     * @param seed the seed
     * @return metric to values
     */
    public static Map<String, double[]> foundation(int n, double[] grid, int perHSteps, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        DoubleFunction<PauliHamiltonian> family = h -> SpinModels.tfim(n, 1, h, true);
        double[] exact = new double[grid.length];
        for (int i = 0; i < grid.length; i++) {
            exact[i] = family.apply(grid[i]).groundState(100, seed).energy();
        }
        // independent RBMs, one per grid coupling
        double[] perH = new double[grid.length];
        long perCost = 0;
        for (int i = 0; i < grid.length; i++) {
            NeuralQuantumState psi = NeuralQuantumState.rbm(n, 1, seed + i);
            VariationalMonteCarlo.Result r = VariationalMonteCarlo.run(family.apply(grid[i]), psi,
                    VariationalMonteCarlo.Method.STOCHASTIC_RECONFIGURATION, perHSteps, 0.05, 1e-3, 0, false, seed);
            perH[i] = rel(r.energy(), exact[i]);
            perCost += perHSteps;
        }
        // one conditional RBM at a quarter of the cost
        int perStep = 4;
        int steps = (int) (perCost / 4 / perStep);
        ConditionalQuantumState c = new ConditionalQuantumState(n, 1, 1.0, seed);
        long cost = c.train(family, 0.2, 2.0, steps, perStep, 0.05, 1e-3, seed);
        double[] cond = new double[grid.length];
        for (int i = 0; i < grid.length; i++) {
            cond[i] = rel(VariationalMonteCarlo.estimate(family.apply(grid[i]), c.at(grid[i]), null, false).energy(),
                    exact[i]);
        }
        // independent RBMs at the conditional model's (quarter) budget
        double[] perHQuarter = new double[grid.length];
        for (int i = 0; i < grid.length; i++) {
            NeuralQuantumState psi = NeuralQuantumState.rbm(n, 1, seed + i);
            VariationalMonteCarlo.Result r = VariationalMonteCarlo.run(family.apply(grid[i]), psi,
                    VariationalMonteCarlo.Method.STOCHASTIC_RECONFIGURATION, Math.max(1, perHSteps / 4), 0.05, 1e-3, 0,
                    false, seed);
            perHQuarter[i] = rel(r.energy(), exact[i]);
        }
        out.put("per-h RBM rel err", perH);
        out.put("per-h RBM at quarter budget rel err", perHQuarter);
        out.put("conditional RBM rel err", cond);
        out.put("mean rel err {per-h, per-h quarter budget, conditional}", new double[] {mean(perH), mean(perHQuarter),
            mean(cond)});
        out.put("cost (coupling-steps) {per-h, conditional}", new double[] {perCost, cost});
        // fidelity susceptibility peak
        double bestModel = 0;
        double bestExact = 0;
        double peakModel = 0;
        double peakExact = 0;
        for (double h = 0.3; h <= 1.81; h += 0.1) {
            double cm = c.fidelitySusceptibility(h);
            double ce = ConditionalQuantumState.exactFidelitySusceptibility(family, h, 0.02);
            if (cm > peakModel) {
                peakModel = cm;
                bestModel = h;
            }
            if (ce > peakExact) {
                peakExact = ce;
                bestExact = h;
            }
        }
        out.put("chi_F peak h {model, exact}", new double[] {bestModel, bestExact});
        out.put("chi_F peak value {model, exact}", new double[] {peakModel, peakExact});
        return out;
    }

    private static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }

    /**
     * Tomography table.
     *
     * @param n qubits for the density-matrix methods
     * @param shotCounts shot budgets
     * @param rbmQubits qubits for the GHZ state of neural tomography
     * @param seed the seed
     * @return metric to values
     */
    public static Map<String, double[]> tomography(int n, int[] shotCounts, int rbmQubits, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        StateVector target = SpinModels.tfim(n, 1, 1, false).groundState(100, seed).state();
        double[] li = new double[shotCounts.length];
        double[] mle = new double[shotCounts.length];
        double[] liNeg = new double[shotCounts.length];
        double[] mleNeg = new double[shotCounts.length];
        for (int i = 0; i < shotCounts.length; i++) {
            Tomography.Shot[] shots = Tomography.measure(target, shotCounts[i], r);
            Tomography.Density a = Tomography.linearInversion(n, shots);
            Tomography.Density b = Tomography.maximumLikelihood(n, shots, 100);
            li[i] = 1 - a.fidelity(target);
            mle[i] = 1 - b.fidelity(target);
            liNeg[i] = a.minEigenvalue();
            mleNeg[i] = b.minEigenvalue();
        }
        out.put("TFIM ground state infidelity linear inversion by shots", li);
        out.put("infidelity MLE (R rho R) by shots", mle);
        out.put("min eigenvalue linear inversion", liNeg);
        out.put("min eigenvalue MLE", mleNeg);
        // neural tomography of GHZ and W
        StateVector ghz = SpinModels.ghz(rbmQubits);
        Tomography.Shot[] gs = Tomography.measure(ghz, 4000, r);
        StateVector rec = Tomography.neuralRbm(rbmQubits, gs, 1, 600, 100, 0.02, seed);
        StateVector w = SpinModels.w(rbmQubits);
        StateVector recW = Tomography.neuralRbm(rbmQubits, Tomography.measure(w, 4000, r), 1, 600, 100, 0.02, seed);
        out.put("neural RBM infidelity {GHZ, W} (" + rbmQubits + " qubits, 4000 shots)",
                new double[] {1 - rec.fidelity(ghz), 1 - recW.fidelity(w)});
        // classical shadows vs direct estimation of all 1- and 2-local Paulis on a 6-qubit state
        int m = 6;
        StateVector psi = SpinModels.tfim(m, 1, 1, false).groundState(100, seed).state();
        List<PauliHamiltonian.Term> obs = Tomography.localPaulis(m);
        int[] budgets = {1000, 3000, 10000};
        double[] shadow = new double[budgets.length];
        double[] direct = new double[budgets.length];
        for (int i = 0; i < budgets.length; i++) {
            Tomography.Shot[] shots = Tomography.measure(psi, budgets[i], r);
            double[] d = Tomography.directEstimate(psi, obs, budgets[i], r);
            for (int k = 0; k < obs.size(); k++) {
                double truth = psi.expectation(obs.get(k).x(), obs.get(k).z());
                shadow[i] = Math.max(shadow[i], Math.abs(Tomography.shadowEstimate(obs.get(k), shots, 5) - truth));
                direct[i] = Math.max(direct[i], Math.abs(d[k] - truth));
            }
        }
        out.put("max error over " + obs.size() + " local Paulis, shots {1e3, 3e3, 1e4}: shadows", shadow);
        out.put("direct per-observable", direct);
        return out;
    }
}
