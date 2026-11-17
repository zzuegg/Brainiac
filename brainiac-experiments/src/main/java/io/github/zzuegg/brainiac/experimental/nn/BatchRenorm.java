package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Batch renormalization: batch norm whose training output is corrected towards the running
 * statistics, {@code xhat = (x - mu_B) / sigma_B * r + d} with
 * {@code r = clip(sigma_B / sigma, 1/rmax, rmax)} and {@code d = clip((mu_B - mu) / sigma, -dmax, dmax)}
 * treated as constants. Training then computes the same function as inference (which uses
 * the running {@code mu, sigma}) whenever the clips are inactive, so small or non-iid batches
 * no longer make the two disagree. {@code rmax} and {@code dmax} are relaxed from 1 and 0 to
 * their limits over a warm-up, starting as plain batch norm.
 *
 * <p>Works on {@code [batch, features]} and {@code [batch, channels, ...]} (statistics over
 * every axis but 1), a drop-in for core {@code BatchNorm}.
 *
 * <p>Paper: Ioffe, "Batch Renormalization: Towards Reducing Minibatch Dependence in
 * Batch-Normalized Models" (NeurIPS 2017), <a href="https://arxiv.org/abs/1702.03275">arXiv:1702.03275</a>.
 */
public final class BatchRenorm implements Layer {

    private final int features;
    private final float momentum;
    private final float rMax;
    private final float dMax;
    private final int warmup;
    private final Var gamma;
    private final Var beta;
    private final Var runningMean;
    private final Var runningVar;
    private int steps;

    /**
     * Creates the layer.
     *
     * @param features the features (channels)
     * @param momentum the running-average rate, such as 0.05
     * @param rMax the final scale clip, such as 3
     * @param dMax the final shift clip, such as 5
     * @param warmup training steps over which the clips relax from (1, 0) to their limits
     */
    public BatchRenorm(int features, float momentum, float rMax, float dMax, int warmup) {
        this.features = features;
        this.momentum = momentum;
        this.rMax = rMax;
        this.dMax = dMax;
        this.warmup = warmup;
        this.gamma = Var.param(Tensor.ones(Shape.of(features)));
        this.beta = Var.param(Tensor.zeros(Shape.of(features)));
        this.runningMean = Var.of(Tensor.zeros(Shape.of(features)));
        this.runningVar = Var.of(Tensor.ones(Shape.of(features)));
    }

    /**
     * Describes the layer with the paper's settings ({@code rmax = 3}, {@code dmax = 5}).
     *
     * @param warmup the warm-up steps
     * @return the spec
     */
    public static LayerSpec spec(int warmup) {
        return TechniqueOps.sameShape((s, r) -> new BatchRenorm(s.dim(0), 0.05f, 3f, 5f, warmup));
    }

    private Shape channelShape(int rank) {
        int[] d = new int[rank];
        java.util.Arrays.fill(d, 1);
        d[1] = features;
        return Shape.of(d);
    }

    /** Moves axis 1 last and flattens the rest: {@code [rows, features]}. */
    private Var rows(Var x) {
        int rank = x.shape().rank();
        if (rank == 2) {
            return x;
        }
        int[] perm = new int[rank];
        perm[rank - 1] = 1;
        for (int i = 0, j = 0; i < rank; i++) {
            if (i != 1) {
                perm[j++] = i;
            }
        }
        long n = x.shape().size() / features;
        return x.permute(perm).reshape(Math.toIntExact(n), features);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().rank() < 2 || input.shape().dim(1) != features) {
            throw new IllegalArgumentException(this + " expects [batch, " + features + ", ...], got " + input.shape());
        }
        Shape cs = channelShape(input.shape().rank());
        Var g = gamma.reshape(cs.toArray());
        Var b = beta.reshape(cs.toArray());
        if (!training) {
            Tensor std = runningVar.value().add(1e-5f).sqrt();
            return input.sub(runningMean.value().reshape(cs.toArray())).div(std.reshape(cs.toArray())).mul(g).add(b);
        }
        Var r2 = rows(input);
        Var mean = r2.mean(0, false);
        Var var = r2.sub(mean).square().mean(0, false);
        Var std = var.add(1e-5f).sqrt();
        float progress = warmup <= 0 ? 1f : Math.min(1f, (float) steps / warmup);
        float rm = 1 + (rMax - 1) * progress;
        float dm = dMax * progress;
        Tensor runStd = runningVar.value().add(1e-5f).sqrt();
        float[] sb = std.value().toArray();
        float[] mb = mean.value().toArray();
        float[] rs = runStd.toArray();
        float[] rmu = runningMean.value().toArray();
        float[] r = new float[features];
        float[] d = new float[features];
        for (int i = 0; i < features; i++) {
            r[i] = Math.max(1 / rm, Math.min(rm, sb[i] / rs[i]));
            d[i] = Math.max(-dm, Math.min(dm, (mb[i] - rmu[i]) / rs[i]));
        }
        Tensor rt = Tensor.of(cs, r).to(input.value().device());
        Tensor dt = Tensor.of(cs, d).to(input.value().device());
        Var xhat = input.sub(mean.reshape(cs.toArray())).div(std.reshape(cs.toArray())).mul(rt).add(dt);
        runningMean.assign(runningMean.value().add(mean.value().sub(runningMean.value()).mul(momentum)));
        runningVar.assign(runningVar.value().add(var.value().sub(runningVar.value()).mul(momentum)));
        steps++;
        return xhat.mul(g).add(b);
    }

    @Override
    public List<Var> parameters() {
        return List.of(gamma, beta);
    }

    @Override
    public List<Var> buffers() {
        return List.of(runningMean, runningVar);
    }

    @Override
    public String toString() {
        return "BatchRenorm(" + features + ")";
    }
}
