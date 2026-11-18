package io.github.zzuegg.brainiac.experiments.science;

import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.experimental.science.DcPowerFlow;
import io.github.zzuegg.brainiac.experimental.science.LearnedOpf;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Learning DC optimal power flow (wave 13, item 13.3.12) on synthetic 9-, 14- and 30-bus grids
 * with load scenarios at ±30% per bus and ±20% system-wide: the LP solve (optimality reference),
 * a direct MLP, DeepOPF (balance completion), DC3 (completion, DC3 loss, inequality
 * correction); a feasibility classifier with counterfactual load repair against LP load
 * shedding; and a WLS state estimate against the raw injection measurements.
 */
public final class PowerGrids {

    private PowerGrids() {
    }

    /**
     * Runs the comparison.
     *
     * @param buses grid sizes
     * @param train training scenarios per grid
     * @param test test scenarios per grid
     * @param steps Adam steps per network
     * @param seed the seed
     * @return rows keyed by grid and method
     */
    public static Map<String, double[]> run(int[] buses, int train, int test, int steps, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int nb : buses) {
            String p = nb + "-bus ";
            ScienceSims.PowerGrid raw = ScienceSims.powerGrid(nb, seed + nb);
            double[][] trLoads = ScienceSims.loadScenarios(raw, train, 0.3, 0.2, seed + 1);
            double[][] teLoads = ScienceSims.loadScenarios(raw, test, 0.3, 0.2, seed + 2);
            // scale the line limits so that about a fifth of the scenarios are infeasible
            double lo = 0.05;
            double hi = 20;
            for (int it = 0; it < 10; it++) {
                double mid = Math.sqrt(lo * hi);
                DcPowerFlow probe = new DcPowerFlow(raw.withLimitsScaled(mid));
                int bad = 0;
                for (int i = 0; i < Math.min(train, 100); i++) {
                    bad += probe.solve(trLoads[i]).feasible() ? 0 : 1;
                }
                if (bad > 0.2 * Math.min(train, 100)) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            ScienceSims.PowerGrid grid = raw.withLimitsScaled(hi);
            DcPowerFlow pf = new DcPowerFlow(grid);
            boolean[] trFeasible = new boolean[train];
            List<double[]> fLoads = new ArrayList<>();
            List<double[]> fDispatch = new ArrayList<>();
            for (int i = 0; i < train; i++) {
                DcPowerFlow.Dispatch d = pf.solve(trLoads[i]);
                trFeasible[i] = d.feasible();
                if (d.feasible()) {
                    fLoads.add(trLoads[i]);
                    fDispatch.add(d.dispatch());
                }
            }
            long t0 = System.nanoTime();
            DcPowerFlow.Dispatch[] lp = new DcPowerFlow.Dispatch[test];
            int feasibleTest = 0;
            for (int i = 0; i < test; i++) {
                lp[i] = pf.solve(teLoads[i]);
                feasibleTest += lp[i].feasible() ? 1 : 0;
            }
            double lpMicros = (System.nanoTime() - t0) / 1e3 / test;
            out.put(p + "LP {feasible share train, test, us per solve}",
                    new double[] {fLoads.size() / (double) train, feasibleTest / (double) test, lpMicros});
            if (fLoads.isEmpty() || feasibleTest == 0) {
                continue;
            }
            double[][] fl = fLoads.toArray(new double[0][]);
            double[][] fd = fDispatch.toArray(new double[0][]);
            double[][] feasibleTe = new double[feasibleTest][];
            DcPowerFlow.Dispatch[] feasibleLp = new DcPowerFlow.Dispatch[feasibleTest];
            for (int i = 0, k = 0; i < test; i++) {
                if (lp[i].feasible()) {
                    feasibleTe[k] = teLoads[i];
                    feasibleLp[k++] = lp[i];
                }
            }
            for (LearnedOpf.Method m : LearnedOpf.Method.values()) {
                LearnedOpf model = new LearnedOpf(m, pf, 64, seed).train(fl, fd, steps, 5e-3);
                long t1 = System.nanoTime();
                double[][] pred = model.predict(feasibleTe);
                double micros = (System.nanoTime() - t1) / 1e3 / feasibleTe.length;
                double gap = 0;
                double viol = 0;
                double mag = 0;
                for (int i = 0; i < feasibleTe.length; i++) {
                    DcPowerFlow.Violation v = pf.violation(pred[i], feasibleTe[i]);
                    double tol = 0.5;
                    if (!v.ok(tol)) {
                        viol++;
                    }
                    mag += v.total() / feasibleTe.length;
                    gap += (pf.cost(pred[i]) - feasibleLp[i].cost()) / feasibleLp[i].cost() / feasibleTe.length;
                }
                out.put(p + m + " {opt gap %, violation rate, mean violation MW, us per scenario}",
                        new double[] {100 * gap, viol / feasibleTe.length, mag, micros});
            }
            // feasibility classification and repair
            LearnedOpf.FeasibilityClassifier clf = new LearnedOpf.FeasibilityClassifier(pf, 32, seed)
                    .train(trLoads, trFeasible, steps, 5e-3);
            double[] logit = clf.logit(teLoads);
            int correct = 0;
            int repaired = 0;
            int infeasible = 0;
            double change = 0;
            double shed = 0;
            for (int i = 0; i < test; i++) {
                correct += (logit[i] > 0) == lp[i].feasible() ? 1 : 0;
                if (lp[i].feasible()) {
                    continue;
                }
                infeasible++;
                double[] r = clf.repair(teLoads[i], 1.0);
                if (r != null && pf.solve(r).feasible()) {
                    repaired++;
                }
                if (r != null) {
                    for (int b = 0; b < r.length; b++) {
                        change += Math.abs(r[b] - teLoads[i][b]);
                    }
                }
                shed += pf.shed(teLoads[i], 1000).totalShed();
            }
            out.put(p + "feasibility classifier {test accuracy, infeasible cases}",
                    new double[] {correct / (double) test, infeasible});
            out.put(p + "repair {counterfactual success, mean |load change| MW, LP shedding success,"
                    + " mean shed MW}", new double[] {infeasible == 0 ? Double.NaN : repaired / (double) infeasible,
                        infeasible == 0 ? Double.NaN : change / infeasible, 1,
                        infeasible == 0 ? Double.NaN : shed / infeasible});
            // WLS state estimation from noisy flows and injections
            SplittableRandom r = new SplittableRandom(seed + 7);
            double rawErr = 0;
            double wlsErr = 0;
            int count = 0;
            for (int i = 0; i < feasibleTe.length; i++) {
                double[] inj = new double[nb];
                for (int g = 0; g < grid.generators(); g++) {
                    inj[grid.genBus()[g]] += feasibleLp[i].dispatch()[g];
                }
                for (int b = 0; b < nb; b++) {
                    inj[b] -= feasibleTe[i][b];
                }
                double[] f = pf.flows(feasibleLp[i].dispatch(), feasibleTe[i]);
                double[] zf = new double[f.length];
                double[] zi = new double[nb];
                for (int l = 0; l < f.length; l++) {
                    zf[l] = f[l] + 2 * r.nextGaussian();
                }
                for (int b = 0; b < nb; b++) {
                    zi[b] = inj[b] + 5 * r.nextGaussian();
                }
                double[] est = pf.estimateInjections(zf, 2, zi, 5);
                for (int b = 1; b < nb; b++) {
                    rawErr += (zi[b] - inj[b]) * (zi[b] - inj[b]);
                    wlsErr += (est[b] - inj[b]) * (est[b] - inj[b]);
                    count++;
                }
            }
            out.put(p + "state estimate {RMSE raw injections, RMSE WLS} MW",
                    new double[] {Math.sqrt(rawErr / count), Math.sqrt(wlsErr / count)});
        }
        return out;
    }
}
