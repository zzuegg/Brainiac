package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** Lion (Chen et al. 2023): the sign of an interpolated momentum, one state tensor. */
final class Lion extends StatefulOptimizer {
    private final float beta1;
    private final float beta2;
    private final float weightDecay;

    Lion(float learningRate, float beta1, float beta2, float weightDecay) {
        super(learningRate, "m");
        if (beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || weightDecay < 0) {
            throw new IllegalArgumentException("bad Lion settings");
        }
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.weightDecay = weightDecay;
    }

    @Override
    Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        float lr = learningRate();
        Tensor mixed = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * weightDecay) : p.value();
        p.assign(w.sub(mixed.sign().mul(lr)));
        Tensor m = s[0] == null ? g.mul(1 - beta2) : s[0].mul(beta2).add(g.mul(1 - beta2));
        return new Tensor[] {m};
    }

    @Override
    public String toString() {
        return "Lion(lr=" + learningRate() + ", weightDecay=" + weightDecay + ")";
    }
}
