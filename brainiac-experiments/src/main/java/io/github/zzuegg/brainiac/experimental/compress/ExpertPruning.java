package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FineGrainedMoe;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Expert pruning for mixtures of experts (Lu et al. 2024, "Not all experts are equal",
 * <a href="https://arxiv.org/abs/2402.14800">arXiv:2402.14800</a>): keep the subset of experts
 * whose layer output on calibration data stays closest to the full layer's, searched over all
 * subsets of the kept size (the paper's enumeration, feasible for the few experts of one layer),
 * and drop the rest together with their router columns. {@link #mostUsed} is the simple baseline
 * that keeps the most frequently routed experts.
 *
 * <p>Works on {@link FineGrainedMoe} layers without shared experts; the reduced layer keeps the
 * kept experts' parameters and router columns, so the softmax renormalises over the survivors.
 */
public final class ExpertPruning {

    private ExpertPruning() {
    }

    /**
     * Searches all subsets of {@code keep} experts.
     *
     * @param moe the layer (no shared experts)
     * @param keep how many experts to keep, at least the layer's top-k
     * @param topK the routed experts per token of the reduced layer
     * @param calibration calibration inputs {@code [rows, features]}
     * @return the indices of the best subset, ascending
     */
    public static int[] search(FineGrainedMoe moe, int keep, int topK, Tensor calibration) {
        int e = moe.routedExperts().size();
        Tensor full = Var.noGrad(() -> moe.forward(Var.of(calibration)).value());
        int[] best = null;
        double bestErr = Double.MAX_VALUE;
        int[] subset = new int[keep];
        for (int i = 0; i < keep; i++) {
            subset[i] = i;
        }
        while (true) {
            FineGrainedMoe reduced = reduce(moe, subset, topK);
            double err = Var.noGrad(() -> reduced.forward(Var.of(calibration)).value().sub(full).square().mean()
                    .item());
            if (err < bestErr) {
                bestErr = err;
                best = subset.clone();
            }
            int i = keep - 1;
            while (i >= 0 && subset[i] == e - keep + i) {
                i--;
            }
            if (i < 0) {
                break;
            }
            subset[i]++;
            for (int j = i + 1; j < keep; j++) {
                subset[j] = subset[j - 1] + 1;
            }
        }
        return best;
    }

    /**
     * Keeps the most frequently routed experts on calibration data.
     *
     * @param moe the layer
     * @param keep how many to keep
     * @param calibration calibration inputs
     * @return the kept indices, ascending
     */
    public static int[] mostUsed(FineGrainedMoe moe, int keep, Tensor calibration) {
        int[] load = usage(moe, calibration);
        Integer[] idx = new Integer[load.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Integer.compare(load[b], load[a]));
        int[] k = new int[keep];
        for (int i = 0; i < keep; i++) {
            k[i] = idx[i];
        }
        Arrays.sort(k);
        return k;
    }

    /**
     * @param moe the layer
     * @param calibration calibration inputs
     * @return how many (row, slot) assignments each expert gets
     */
    public static int[] usage(FineGrainedMoe moe, Tensor calibration) {
        Var.noGrad(() -> moe.forward(Var.of(calibration)).value());
        return moe.expertLoad();
    }

    /**
     * The layer restricted to some experts.
     *
     * @param moe the layer (no shared experts)
     * @param kept the experts to keep
     * @param topK the routed experts per token of the result
     * @return a new layer sharing the kept experts' parameters, with copied router columns
     */
    public static FineGrainedMoe reduce(FineGrainedMoe moe, int[] kept, int topK) {
        List<Layer> experts = new ArrayList<>();
        for (int i : kept) {
            experts.add(moe.routedExperts().get(i));
        }
        return withRouter(moe.router().value(), kept, experts, topK);
    }

    /**
     * A layer from given experts and the matching columns of a router.
     *
     * @param router the full router {@code [features, experts]}
     * @param columns the router columns, one per expert in order
     * @param experts the experts
     * @param topK the routed experts per token
     * @return the layer
     */
    public static FineGrainedMoe withRouter(Tensor router, int[] columns, List<? extends Layer> experts, int topK) {
        int features = router.shape().dim(0);
        int e = router.shape().dim(1);
        float[] r = router.toArray();
        float[] sub = new float[features * columns.length];
        for (int f = 0; f < features; f++) {
            for (int j = 0; j < columns.length; j++) {
                sub[f * columns.length + j] = r[f * e + columns[j]];
            }
        }
        FineGrainedMoe out = new FineGrainedMoe(features, features, Math.min(topK, experts.size()), experts,
                List.of(), new SplittableRandom(0));
        out.router().assign(Tensor.of(Shape.of(features, columns.length), sub));
        return out;
    }
}
