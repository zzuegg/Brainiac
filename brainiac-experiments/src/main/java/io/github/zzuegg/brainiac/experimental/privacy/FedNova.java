package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * FedNova's normalised averaging (Wang et al., NeurIPS 2020,
 * <a href="https://arxiv.org/abs/2007.07481">arXiv:2007.07481</a>). When clients take
 * different numbers of local steps {@code τ_i}, plain weighted averaging of their updates
 * optimises a different objective, weighted towards the clients that worked most. FedNova
 * divides each update by its step count (an average per-step direction {@code d_i = Δ_i / τ_i}),
 * averages those with the data weights {@code p_i}, and rescales by the effective step count
 * {@code τ_eff = Σ p_i τ_i}: {@code Δ = τ_eff Σ p_i d_i}. With equal {@code τ_i} it is plain
 * FedAvg.
 */
public final class FedNova {

    private FedNova() {
    }

    /**
     * Aggregates client updates.
     *
     * @param updates {@code [client][parameter]} updates {@code w_i - w}
     * @param weights the data weight of each client (normalised here)
     * @param steps the local steps each client took
     * @return the normalised aggregate update
     */
    public static Tensor[] aggregate(Tensor[][] updates, double[] weights, int[] steps) {
        double total = 0;
        for (double w : weights) {
            total += w;
        }
        double tauEff = 0;
        for (int i = 0; i < weights.length; i++) {
            tauEff += weights[i] / total * steps[i];
        }
        Tensor[] out = new Tensor[updates[0].length];
        for (int i = 0; i < updates.length; i++) {
            float scale = (float) (tauEff * weights[i] / total / Math.max(1, steps[i]));
            for (int p = 0; p < out.length; p++) {
                Tensor t = updates[i][p].mul(scale);
                out[p] = out[p] == null ? t : out[p].add(t);
            }
        }
        return out;
    }

    /**
     * Plain weighted averaging, FedAvg's aggregate, for comparison.
     *
     * @param updates {@code [client][parameter]} updates
     * @param weights the data weight of each client
     * @return the weighted mean update
     */
    public static Tensor[] average(Tensor[][] updates, double[] weights) {
        double total = 0;
        for (double w : weights) {
            total += w;
        }
        Tensor[] out = new Tensor[updates[0].length];
        for (int i = 0; i < updates.length; i++) {
            float scale = (float) (weights[i] / total);
            for (int p = 0; p < out.length; p++) {
                Tensor t = updates[i][p].mul(scale);
                out[p] = out[p] == null ? t : out[p].add(t);
            }
        }
        return out;
    }
}
