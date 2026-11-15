package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/** The Adam optimizer. */
final class Adam implements Optimizer {
    private float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final Map<Var, Tensor> m = new IdentityHashMap<>();
    private final Map<Var, Tensor> v = new IdentityHashMap<>();
    private int t;

    Adam(float learningRate, float beta1, float beta2, float epsilon) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive, got " + learningRate);
        }
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public void step(List<Var> parameters) {
        t++;
        float correction1 = 1 - (float) Math.pow(beta1, t);
        float correction2 = 1 - (float) Math.pow(beta2, t);
        for (Var p : parameters) {
            Tensor g = p.grad();
            if (g == null) {
                continue;
            }
            Tensor mPrev = m.get(p);
            Tensor vPrev = v.get(p);
            Tensor mNew = (mPrev == null ? g.mul(1 - beta1) : mPrev.mul(beta1).add(g.mul(1 - beta1)));
            Tensor vNew = (vPrev == null ? g.square().mul(1 - beta2) : vPrev.mul(beta2).add(g.square().mul(1 - beta2)));
            m.put(p, mNew);
            v.put(p, vNew);
            Tensor mHat = mNew.div(correction1);
            Tensor vHat = vNew.div(correction2);
            p.assign(p.value().sub(mHat.div(vHat.sqrt().add(epsilon)).mul(learningRate)));
            p.zeroGrad();
        }
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive, got " + learningRate);
        }
        this.learningRate = learningRate;
    }

    @Override
    public String toString() {
        return "Adam(lr=" + learningRate + ")";
    }
}
