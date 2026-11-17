package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** Adam, and AdamW when it has a (decoupled) weight decay. */
final class Adam extends StatefulOptimizer {
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;

    Adam(float learningRate, float beta1, float beta2, float epsilon) {
        this(learningRate, beta1, beta2, epsilon, 0);
    }

    Adam(float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
        super(learningRate, "m", "v");
        if (beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || epsilon < 0 || weightDecay < 0) {
            throw new IllegalArgumentException("bad Adam settings");
        }
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }

    @Override
    Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        float lr = learningRate();
        float correction1 = 1 - (float) Math.pow(beta1, t);
        float correction2 = 1 - (float) Math.pow(beta2, t);
        Tensor m = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor v = s[1] == null ? g.square().mul(1 - beta2) : s[1].mul(beta2).add(g.square().mul(1 - beta2));
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * weightDecay) : p.value();
        p.assign(w.sub(m.div(correction1).div(v.div(correction2).sqrt().add(epsilon)).mul(lr)));
        return new Tensor[] {m, v};
    }

    @Override
    public String toString() {
        return (weightDecay > 0 ? "AdamW(lr=" + learningRate() + ", weightDecay=" + weightDecay : "Adam(lr="
                + learningRate()) + ")";
    }
}
