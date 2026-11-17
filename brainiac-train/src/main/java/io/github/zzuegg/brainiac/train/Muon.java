package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Muon (Jordan et al. 2024), as PyTorch implements it: momentum whose update is
 * orthogonalized by five Newton-Schulz steps. Only for matrices; give the other parameters
 * (biases, norms, embeddings) to AdamW with {@link Optimizer#groups}.
 */
final class Muon extends StatefulOptimizer {
    private static final float A = 3.4445f;
    private static final float B = -4.775f;
    private static final float C = 2.0315f;
    private static final float EPS = 1e-7f;

    private final float weightDecay;
    private final float momentum;
    private final boolean nesterov;
    private final boolean matchAdamW;
    private final int nsSteps;

    Muon(float learningRate, float weightDecay, float momentum, boolean nesterov, boolean matchAdamW, int nsSteps) {
        super(learningRate, "momentum");
        if (weightDecay < 0 || momentum < 0 || momentum >= 1 || nsSteps < 1) {
            throw new IllegalArgumentException("bad Muon settings");
        }
        this.weightDecay = weightDecay;
        this.momentum = momentum;
        this.nesterov = nesterov;
        this.matchAdamW = matchAdamW;
        this.nsSteps = nsSteps;
    }

    /** The orthogonal-ish matrix nearest {@code g}: U S' V^T with S' near 1. */
    static Tensor orthogonalize(Tensor g, int steps) {
        boolean tall = g.shape().dim(0) > g.shape().dim(1);
        Tensor x = tall ? g.transpose() : g;
        Tensor norm = x.square().sum().sqrt();
        x = x.div(norm.maximum(Tensor.full(norm.shape(), EPS, norm.dtype(), norm.device())));
        for (int i = 0; i < steps; i++) {
            Tensor a = x.matmul(x.transpose());
            Tensor b = a.mul(B).add(a.matmul(a).mul(C));
            x = x.mul(A).add(b.matmul(x));
        }
        return tall ? x.transpose() : x;
    }

    @Override
    Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        if (g.rank() != 2) {
            throw new IllegalArgumentException("Muon updates matrices only, got a parameter of shape " + g.shape()
                    + "; use Optimizer.groups to give the rest to AdamW");
        }
        float lr = learningRate();
        Tensor buf = s[0] == null ? g.mul(1 - momentum) : s[0].mul(momentum).add(g.mul(1 - momentum));
        Tensor u = nesterov ? g.mul(1 - momentum).add(buf.mul(momentum)) : buf;
        u = orthogonalize(u, nsSteps);
        int rows = g.shape().dim(0);
        int cols = g.shape().dim(1);
        double ratio = matchAdamW ? 0.2 * Math.sqrt(Math.max(rows, cols)) : Math.sqrt(Math.max(1.0, rows / (double) cols));
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * weightDecay) : p.value();
        p.assign(w.sub(u.mul((float) (lr * ratio))));
        return new Tensor[] {buf};
    }

    @Override
    public String toString() {
        return "Muon(lr=" + learningRate() + ", weightDecay=" + weightDecay + ")";
    }
}
