package io.github.zzuegg.brainiac.experiments.science;

import io.github.zzuegg.brainiac.experimental.ml.LogDensity;
import io.github.zzuegg.brainiac.experimental.science.FlowSampler;
import io.github.zzuegg.brainiac.experimental.science.LatticePhi4;
import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Flow-based Markov chain Monte Carlo for two-dimensional lattice {@code phi^4} theory (wave
 * 13, item 13.3.6): a checkerboard RealNVP flow trained by reverse KL proposes whole
 * configurations to an independence Metropolis–Hastings chain, against local Metropolis
 * sweeps, Hamiltonian Monte Carlo on the field, and importance sampling from a Gaussian.
 * Reported per coupling: the integrated autocorrelation time of {@code |M|} (per sweep, per
 * trajectory, per proposal), the flow's acceptance rate and the effective sample size per
 * sample of its weights, and {@code <|M|>} from the local chain and the flow chain.
 */
public final class LatticeSampling {

    private LatticeSampling() {
    }

    /**
     * Runs the comparison on one lattice.
     *
     * @param side the lattice side
     * @param lambdas quartic couplings (with {@code m^2 = -4})
     * @param flowSteps reverse-KL training steps
     * @param chain chain length for every sampler
     * @param seed the seed
     * @return metric to values (one per coupling)
     */
    public static Map<String, double[]> run(int side, double[] lambdas, int flowSteps, int chain, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] keys = {"tau_int |M| local Metropolis (per sweep)", "tau_int |M| HMC (per trajectory)",
            "tau_int |M| flow IMH (per proposal)", "flow acceptance", "flow ESS per sample",
            "Gaussian IS ESS per sample", "<|M|> local", "<|M|> flow", "<|M|> HMC", "seconds {local, HMC, flow incl. training}"};
        for (String k : keys) {
            out.put(k, new double[keys[9].equals(k) ? 3 * lambdas.length : lambdas.length]);
        }
        for (int i = 0; i < lambdas.length; i++) {
            LatticePhi4 theory = new LatticePhi4(side, -4, lambdas[i]);
            int v = theory.volume();
            SplittableRandom r = new SplittableRandom(seed + i);
            // local Metropolis
            long t0 = System.nanoTime();
            double[] phi = new double[v];
            theory.metropolis(phi, 300, 1.0, r);
            double[] mLocal = new double[chain];
            for (int s = 0; s < chain; s++) {
                theory.metropolis(phi, 1, 1.0, r);
                mLocal[s] = Math.abs(LatticePhi4.magnetization(phi));
            }
            long t1 = System.nanoTime();
            // HMC
            LogDensity target = theory.logDensity();
            HamiltonianMonteCarlo.Result h = HamiltonianMonteCarlo.hmc(target, phi, 0, 10, 200, chain / 2, 0.8, r);
            double[] mHmc = new double[h.samples().length];
            for (int s = 0; s < mHmc.length; s++) {
                mHmc[s] = Math.abs(LatticePhi4.magnetization(h.samples()[s]));
            }
            long t2 = System.nanoTime();
            // flow
            FlowSampler flow = new FlowSampler(theory, 6, 64, seed + i);
            flow.train(flowSteps, 64, 3e-3, seed + i);
            FlowSampler.Chain c = flow.sample(chain, r);
            double[] mFlow = new double[chain];
            for (int s = 0; s < chain; s++) {
                mFlow[s] = Math.abs(c.magnetizations()[s]);
            }
            long t3 = System.nanoTime();
            // Gaussian importance sampling with the local chain's field scale
            double var = 0;
            for (double p : phi) {
                var += p * p / v;
            }
            double sd = Math.sqrt(Math.max(var, 0.1));
            double[] lw = new double[chain];
            double max = Double.NEGATIVE_INFINITY;
            for (int s = 0; s < chain; s++) {
                double[] x = new double[v];
                double lq = 0;
                for (int k = 0; k < v; k++) {
                    double z = r.nextGaussian();
                    x[k] = sd * z;
                    lq -= 0.5 * z * z;
                }
                lw[s] = -theory.action(x) - lq;
                max = Math.max(max, lw[s]);
            }
            double sw = 0;
            double sw2 = 0;
            for (double l : lw) {
                double w = Math.exp(l - max);
                sw += w;
                sw2 += w * w;
            }
            out.get(keys[0])[i] = LatticePhi4.integratedAutocorrelation(mLocal);
            out.get(keys[1])[i] = LatticePhi4.integratedAutocorrelation(mHmc);
            out.get(keys[2])[i] = LatticePhi4.integratedAutocorrelation(mFlow);
            out.get(keys[3])[i] = c.acceptance();
            out.get(keys[4])[i] = c.essPerSample();
            out.get(keys[5])[i] = sw * sw / sw2 / chain;
            out.get(keys[6])[i] = mean(mLocal);
            out.get(keys[7])[i] = mean(mFlow);
            out.get(keys[8])[i] = mean(mHmc);
            out.get(keys[9])[3 * i] = (t1 - t0) / 1e9;
            out.get(keys[9])[3 * i + 1] = (t2 - t1) / 1e9;
            out.get(keys[9])[3 * i + 2] = (t3 - t2) / 1e9;
        }
        return out;
    }

    private static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }
}
