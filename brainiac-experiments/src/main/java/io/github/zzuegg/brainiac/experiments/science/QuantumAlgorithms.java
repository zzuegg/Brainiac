package io.github.zzuegg.brainiac.experiments.science;

import io.github.zzuegg.brainiac.experimental.combopt.ClassicHeuristics;
import io.github.zzuegg.brainiac.experimental.quantum.PauliHamiltonian;
import io.github.zzuegg.brainiac.experimental.quantum.Qaoa;
import io.github.zzuegg.brainiac.experimental.quantum.QuantumCircuit;
import io.github.zzuegg.brainiac.experimental.quantum.SpinModels;
import io.github.zzuegg.brainiac.experimental.quantum.Vqe;
import io.github.zzuegg.brainiac.experiments.exotic.ProbabilisticBits;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Variational quantum algorithms on exactly simulated state vectors (wave 13, items 13.3.1 and
 * 13.3.2).
 *
 * <ul>
 *   <li>{@link #vqe}: the variational quantum eigensolver on the transverse-field Ising chain at
 *       several fields and on the XXZ chain: relative energy error and ground-state fidelity of
 *       the Hamiltonian variational ansatz, the hardware-efficient ansatz (the circuit of
 *       {@code QuantumCircuitLayer}) and the mean-field product state against exact Lanczos
 *       energies, gradient variance at random initialisation (barren plateaus), SPSA with
 *       finite shots, and agreement of adjoint and parameter-shift gradients.</li>
 *   <li>{@link #qaoa}: QAOA for MaxCut on random 3-regular graphs: approximation ratio and
 *       optimal-cut probability against depth for random, INTERP and Fourier initialisations,
 *       transfer of angles from a smaller graph, and the classical Ising-machine and local
 *       search baselines on the same instance.</li>
 * </ul>
 */
public final class QuantumAlgorithms {

    private QuantumAlgorithms() {
    }

    /**
     * VQE table.
     *
     * @param n chain length
     * @param fields the transverse fields {@code h / J}
     * @param layers ansatz depth
     * @param seed the seed
     * @return metric to values (one per field unless stated)
     */
    public static Map<String, double[]> vqe(int n, double[] fields, int layers, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        int k = fields.length;
        String[] keys = {"rel err HVA (L-BFGS)", "rel err HEA (L-BFGS)", "rel err HEA (Adam)",
            "rel err mean field", "fidelity HVA", "fidelity HEA", "params HVA", "params HEA"};
        for (String key : keys) {
            out.put(key, new double[k]);
        }
        for (int i = 0; i < k; i++) {
            PauliHamiltonian h = SpinModels.tfim(n, 1, fields[i], true);
            PauliHamiltonian.GroundState gs = h.groundState(120, seed);
            double e0 = gs.energy();
            Vqe hva = new Vqe(Vqe.tfimVariational(h, layers), h);
            Vqe.Result rh = hva.bestOf(3, 200, 0.1, seed);
            Vqe hea = new Vqe(Vqe.hardwareEfficient(n, layers), h);
            Vqe.Result re = hea.bestOf(3, 200, 0.1, seed);
            Vqe heaAdam = new Vqe(Vqe.hardwareEfficient(n, layers), h);
            Vqe.Result ra = heaAdam.optimize(Vqe.Method.ADAM, heaAdam.initial(0.1, seed), 300, 0.05, 0, 0);
            Vqe mf = new Vqe(Vqe.product(n), h);
            Vqe.Result rm = mf.bestOf(3, 200, 1.0, seed);
            out.get("rel err HVA (L-BFGS)")[i] = rel(rh.energy(), e0);
            out.get("rel err HEA (L-BFGS)")[i] = rel(re.energy(), e0);
            out.get("rel err HEA (Adam)")[i] = rel(ra.energy(), e0);
            out.get("rel err mean field")[i] = rel(rm.energy(), e0);
            out.get("fidelity HVA")[i] = hva.fidelity(rh.parameters(), gs.state());
            out.get("fidelity HEA")[i] = hea.fidelity(re.parameters(), gs.state());
            out.get("params HVA")[i] = hva.circuit().parameterCount();
            out.get("params HEA")[i] = hea.circuit().parameterCount();
        }
        // barren plateaus: variance of the first derivative over random angles
        PauliHamiltonian crit = SpinModels.tfim(n, 1, 1, true);
        out.put("grad variance {HVA, HEA}", new double[] {
            new Vqe(Vqe.tfimVariational(crit, layers), crit).gradientVariance(0, 100, seed),
            new Vqe(Vqe.hardwareEfficient(n, layers), crit).gradientVariance(0, 100, seed)});
        // SPSA with 1000 shots per term on the HVA at the critical point
        double e0 = crit.groundState(120, seed).energy();
        Vqe spsa = new Vqe(Vqe.tfimVariational(crit, layers), crit);
        Vqe.Result rs = spsa.optimize(Vqe.Method.SPSA, spsa.initial(0.1, seed), 400, 0.2, 1000, seed);
        out.put("SPSA 1000 shots HVA {rel err, evaluations}", new double[] {rel(rs.energy(), e0), rs.evaluations()});
        // adjoint against parameter shift on the hardware-efficient circuit
        QuantumCircuit c = Vqe.hardwareEfficient(n, layers);
        double[] t = new Vqe(c, crit).initial(1.0, seed);
        double[] adj = new double[t.length];
        c.energy(crit, t, adj);
        double[] ps = c.parameterShift(crit, t);
        double diff = 0;
        for (int j = 0; j < t.length; j++) {
            diff = Math.max(diff, Math.abs(adj[j] - ps[j]));
        }
        out.put("max |adjoint - parameter shift|", new double[] {diff});
        // XXZ chain with the Hamiltonian variational ansatz
        PauliHamiltonian xxz = SpinModels.xxz(n, 1, 1, true);
        double ex = xxz.groundState(120, seed).energy();
        Vqe xh = new Vqe(Vqe.xxzVariational(n, true, layers), xxz);
        Vqe xe = new Vqe(Vqe.hardwareEfficient(n, layers), xxz);
        out.put("XXZ rel err {HVA, HEA}", new double[] {rel(xh.bestOf(3, 200, 0.1, seed).energy(), ex),
            rel(xe.bestOf(3, 200, 0.1, seed).energy(), ex)});
        return out;
    }

    private static double rel(double e, double e0) {
        return Math.abs((e - e0) / e0);
    }

    /**
     * QAOA table.
     *
     * @param n nodes of the test graph (3-regular)
     * @param smallN nodes of the graph whose angles are transferred
     * @param maxP the largest depth
     * @param budget Nelder–Mead iterations per depth
     * @param seed the seed
     * @return metric to values
     */
    public static Map<String, double[]> qaoa(int n, int smallN, int maxP, int budget, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] w = SpinModels.regularGraph(n, 3, seed);
        Qaoa q = new Qaoa(w);
        for (Qaoa.Init init : Qaoa.Init.values()) {
            List<Qaoa.Result> res = q.sweep(init, maxP, budget, seed);
            double[] ratio = new double[maxP];
            double[] popt = new double[maxP];
            double evals = 0;
            for (int p = 0; p < maxP; p++) {
                ratio[p] = res.get(p).ratio();
                popt[p] = res.get(p).optimalProbability();
                evals += res.get(p).evaluations();
            }
            out.put(init + " ratio by p", ratio);
            out.put(init + " P(optimal) by p", popt);
            out.put(init + " evaluations", new double[] {evals});
        }
        // transfer: angles optimised on a smaller 3-regular graph
        int pt = Math.min(3, maxP);
        Qaoa small = new Qaoa(SpinModels.regularGraph(smallN, 3, seed + 7));
        Qaoa.Result src = small.sweep(Qaoa.Init.INTERP, pt, budget, seed).get(pt - 1);
        double transferred = q.expectation(src.gammas(), src.betas()) / q.maxCut();
        double reopt = q.sweep(Qaoa.Init.INTERP, pt, budget, seed).get(pt - 1).ratio();
        out.put("transfer p=" + pt + " {transferred, re-optimised}", new double[] {transferred, reopt});
        // classical baselines
        double best = q.maxCut();
        double sb = ProbabilisticBits.ising(w, true, ProbabilisticBits.IsingSolver.SB_BALLISTIC, 5, 500, seed).best();
        double pbit = ProbabilisticBits.ising(w, true, ProbabilisticBits.IsingSolver.PBIT_ANNEALING, 5, 500, seed)
                .best();
        double greedy = ClassicHeuristics.cut(w, ClassicHeuristics.greedyMaxCut(w));
        SplittableRandom r = new SplittableRandom(seed);
        double rnd = 0;
        for (int i = 0; i < 200; i++) {
            int[] side = new int[n];
            for (int j = 0; j < n; j++) {
                side[j] = r.nextInt(2);
            }
            rnd += ClassicHeuristics.cut(w, side);
        }
        out.put("classical ratio {SB, p-bit, greedy+LS, random}", new double[] {sb / best, pbit / best,
            greedy / best, rnd / 200 / best});
        return out;
    }
}
