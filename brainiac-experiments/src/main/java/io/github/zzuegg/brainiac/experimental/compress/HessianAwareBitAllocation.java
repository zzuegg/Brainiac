package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.Supplier;

/**
 * Hessian-aware mixed-precision quantisation, HAWQ-V2 (Dong et al. 2020,
 * <a href="https://arxiv.org/abs/1911.03852">arXiv:1911.03852</a>; HAWQ, <a href="https://arxiv.org/abs/1905.03696">arXiv:1905.03696</a>).
 * Layers differ in how much quantisation noise the loss tolerates. HAWQ-V2 measures layer
 * {@code l}'s sensitivity to {@code b}-bit quantisation as
 *
 * <pre>
 * Omega_l(b) = Tr(H_l) / n_l * ||Q_b(W_l) - W_l||^2
 * </pre>
 *
 * with the average Hessian trace estimated by Hutchinson's method, {@code Tr(H) = E[v^T H v]} for
 * Rademacher {@code v} ({@link #traces}, by exact Hessian-vector products from
 * {@link CurvatureProducts#hessianVector}), and picks one bit width per layer minimising the total
 * sensitivity under a model-size budget ({@link #allocate}, an exhaustive search of the Pareto
 * frontier, cheap for a few layers). {@link #weightVariances} gives the allocation-by-variance
 * baseline when used in place of the traces.
 */
public final class HessianAwareBitAllocation {

    private HessianAwareBitAllocation() {
    }

    /**
     * Hutchinson estimates of every layer's average Hessian trace {@code Tr(H_l) / n_l}.
     *
     * @param loss the loss at the current weights
     * @param weights the layers' weights
     * @param probes Rademacher probes, for example 32
     * @param seed the seed
     * @return one average trace per layer
     */
    public static double[] traces(Supplier<Var> loss, List<Var> weights, int probes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int total = CurvatureProducts.size(weights);
        double[] t = new double[weights.size()];
        for (int p = 0; p < probes; p++) {
            double[] v = new double[total];
            for (int i = 0; i < total; i++) {
                v[i] = r.nextBoolean() ? 1 : -1;
            }
            double[] hv = CurvatureProducts.hessianVector(loss, weights, v);
            int o = 0;
            for (int l = 0; l < weights.size(); l++) {
                int n = (int) weights.get(l).shape().size();
                double s = 0;
                for (int i = o; i < o + n; i++) {
                    s += v[i] * hv[i];
                }
                t[l] += s / n / probes;
                o += n;
            }
        }
        return t;
    }

    /**
     * @param weights the layers' weights
     * @return each layer's weight variance (the baseline's sensitivity proxy)
     */
    public static double[] weightVariances(List<Var> weights) {
        double[] out = new double[weights.size()];
        for (int l = 0; l < out.length; l++) {
            float[] v = weights.get(l).value().toArray();
            double mean = 0;
            for (float f : v) {
                mean += f / (double) v.length;
            }
            double var = 0;
            for (float f : v) {
                var += (f - mean) * (f - mean) / v.length;
            }
            out[l] = var;
        }
        return out;
    }

    /**
     * {@code Omega_l(b)} for every layer and candidate bit width.
     *
     * @param scale the per-layer factor (average Hessian trace, or a proxy)
     * @param weights the layers' weights
     * @param bits the candidate bit widths
     * @param quantizer quantises a weight to a bit width
     * @return {@code [layers][candidates]}
     */
    public static double[][] sensitivities(double[] scale, List<Var> weights, int[] bits,
            BiFunction<Tensor, Integer, Tensor> quantizer) {
        double[][] omega = new double[weights.size()][bits.length];
        for (int l = 0; l < weights.size(); l++) {
            Tensor w = weights.get(l).value();
            for (int b = 0; b < bits.length; b++) {
                double err = quantizer.apply(w, bits[b]).sub(w).square().sum().item();
                omega[l][b] = scale[l] * err;
            }
        }
        return omega;
    }

    /**
     * The bit width per layer minimising the total sensitivity within a size budget.
     *
     * @param omega {@code [layers][candidates]} sensitivities
     * @param bits the candidate bit widths
     * @param sizes the weights per layer
     * @param averageBits the budget, as average bits per weight
     * @return the chosen bit width per layer
     */
    public static int[] allocate(double[][] omega, int[] bits, long[] sizes, double averageBits) {
        int layers = omega.length;
        long total = 0;
        for (long s : sizes) {
            total += s;
        }
        double budget = averageBits * total + 1e-9;
        int[] choice = new int[layers];
        int[] best = null;
        double bestCost = Double.MAX_VALUE;
        long combos = (long) Math.pow(bits.length, layers);
        for (long c = 0; c < combos; c++) {
            long x = c;
            double size = 0;
            double cost = 0;
            for (int l = 0; l < layers; l++) {
                choice[l] = (int) (x % bits.length);
                x /= bits.length;
                size += (double) bits[choice[l]] * sizes[l];
                cost += omega[l][choice[l]];
            }
            if (size <= budget && cost < bestCost) {
                bestCost = cost;
                best = new int[layers];
                for (int l = 0; l < layers; l++) {
                    best[l] = bits[choice[l]];
                }
            }
        }
        if (best == null) {
            throw new IllegalArgumentException("no allocation fits " + averageBits + " bits on average");
        }
        return best;
    }

    /**
     * A random allocation within the budget, the control.
     *
     * @param bits the candidate bit widths
     * @param sizes the weights per layer
     * @param averageBits the budget
     * @param seed the seed
     * @return a random feasible bit width per layer, using most of the budget
     */
    public static int[] random(int[] bits, long[] sizes, double averageBits, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] noise = new double[sizes.length][bits.length];
        // random per-layer sensitivities that still prefer more bits, so the budget is used
        for (double[] row : noise) {
            double scale = Math.exp(4 * r.nextGaussian());
            for (int b = 0; b < bits.length; b++) {
                row[b] = scale * Math.pow(4, -bits[b]);
            }
        }
        return allocate(noise, bits, sizes, averageBits);
    }

    /**
     * @param bitsPerLayer the chosen widths
     * @param sizes the weights per layer
     * @return the average bits per weight
     */
    public static double averageBits(int[] bitsPerLayer, long[] sizes) {
        double s = 0;
        double n = 0;
        for (int l = 0; l < sizes.length; l++) {
            s += (double) bitsPerLayer[l] * sizes[l];
            n += sizes[l];
        }
        return s / n;
    }
}
