package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Adafactor (Shazeer and Stern 2018), as transformers implements it: for a matrix, the
 * squared-gradient average is kept as one row and one column instead of a full copy, so the
 * state is tiny. Updates are clipped by their root mean square, and with {@code
 * scaleParameter} the step is relative to the size of the weights.
 */
final class Adafactor extends StatefulOptimizer {
    private static final float EPS1 = 1e-30f;
    private static final float EPS2 = 1e-3f;
    private static final float CLIP = 1;
    private static final double DECAY = -0.8;

    private final boolean relativeStep;
    private final boolean scaleParameter;
    private final float weightDecay;

    Adafactor(float learningRate, boolean relativeStep, boolean scaleParameter, float weightDecay) {
        super(learningRate, "row", "col", "v");
        if (weightDecay < 0) {
            throw new IllegalArgumentException("weightDecay < 0");
        }
        this.relativeStep = relativeStep;
        this.scaleParameter = scaleParameter;
        this.weightDecay = weightDecay;
    }

    private static Tensor rms(Tensor x) {
        return x.square().mean().sqrt();
    }

    @Override
    Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor w = p.value();
        float step = relativeStep ? (float) Math.min(1e-2, 1 / Math.sqrt(t)) : learningRate();
        Tensor lr = scaleParameter ? rms(w).maximum(Tensor.full(rms(w).shape(), EPS2, w.dtype(), w.device())).mul(step)
                : null;
        float beta = (float) (1 - Math.pow(t, DECAY));
        Tensor sq = g.square().add(EPS1);
        Tensor[] next = new Tensor[3];
        Tensor u;
        int r = g.rank();
        if (r >= 2) {
            Tensor row = sq.mean(r - 1, true);   // [..., rows, 1]
            Tensor col = sq.mean(r - 2, true);   // [..., 1, cols]
            row = s[0] == null ? row.mul(1 - beta) : s[0].mul(beta).add(row.mul(1 - beta));
            col = s[1] == null ? col.mul(1 - beta) : s[1].mul(beta).add(col.mul(1 - beta));
            next[0] = row;
            next[1] = col;
            Tensor rowFactor = row.div(row.mean(r - 2, true)).sqrt();
            u = g.div(rowFactor).div(col.sqrt());
        } else {
            Tensor v = s[2] == null ? sq.mul(1 - beta) : s[2].mul(beta).add(sq.mul(1 - beta));
            next[2] = v;
            u = g.div(v.sqrt());
        }
        Tensor clip = rms(u).div(CLIP);
        u = u.div(clip.maximum(Tensor.full(clip.shape(), 1, clip.dtype(), clip.device())));
        u = lr != null ? u.mul(lr) : u.mul(step);
        if (weightDecay > 0) {
            w = lr != null ? w.sub(w.mul(lr).mul(weightDecay)) : w.mul(1 - weightDecay * step);
        }
        p.assign(w.sub(u));
        return next;
    }

    @Override
    public String toString() {
        return "Adafactor(" + (relativeStep ? "relative step" : "lr=" + learningRate()) + ")";
    }
}
