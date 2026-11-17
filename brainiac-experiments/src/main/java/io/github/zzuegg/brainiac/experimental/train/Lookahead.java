package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Lookahead (Zhang, Lucas, Hinton and Ba 2019, <a href="https://arxiv.org/abs/1907.08610">arXiv:1907.08610</a>):
 * wraps any optimizer, whose "fast" weights explore for {@code k} steps; then the "slow"
 * weights move a fraction {@code alpha} towards them and the fast weights restart from the
 * slow ones: {@code slow += alpha (fast - slow); fast = slow}. Reduces variance and makes
 * the inner optimizer less sensitive to its learning rate.
 */
public final class Lookahead implements Optimizer {

    private final Optimizer inner;
    private final int k;
    private final float alpha;
    private final Map<Var, Tensor> slow = new IdentityHashMap<>();
    private int steps;

    /**
     * @param inner the fast optimizer
     * @param k fast steps per slow step, for example 5
     * @param alpha the slow step fraction, for example 0.5
     */
    public Lookahead(Optimizer inner, int k, float alpha) {
        this.inner = inner;
        this.k = k;
        this.alpha = alpha;
    }

    @Override
    public void step(List<Var> parameters) {
        for (Var p : parameters) {
            slow.computeIfAbsent(p, q -> q.value());
        }
        inner.step(parameters);
        steps++;
        if (steps % k == 0) {
            for (Var p : parameters) {
                Tensor s = slow.get(p);
                Tensor updated = s.add(p.value().sub(s).mul(alpha));
                slow.put(p, updated);
                p.assign(updated);
            }
        }
    }

    @Override
    public float learningRate() {
        return inner.learningRate();
    }

    @Override
    public void learningRate(float learningRate) {
        inner.learningRate(learningRate);
    }

    @Override
    public Map<String, Tensor> state(List<Var> parameters) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        inner.state(parameters).forEach((key, v) -> out.put("inner." + key, v));
        out.put("lookahead.step", Tensor.of(steps));
        for (int i = 0; i < parameters.size(); i++) {
            Tensor s = slow.get(parameters.get(i));
            if (s != null) {
                out.put(i + ".slow", s);
            }
        }
        return out;
    }

    @Override
    public void loadState(List<Var> parameters, Map<String, Tensor> state) {
        Map<String, Tensor> in = new LinkedHashMap<>();
        state.forEach((key, v) -> {
            if (key.startsWith("inner.")) {
                in.put(key.substring(6), v);
            }
        });
        inner.loadState(parameters, in);
        Tensor s = state.get("lookahead.step");
        steps = s == null ? 0 : Math.round(s.item());
        slow.clear();
        for (int i = 0; i < parameters.size(); i++) {
            Tensor x = state.get(i + ".slow");
            if (x != null) {
                slow.put(parameters.get(i), x);
            }
        }
    }

    @Override
    public String toString() {
        return "Lookahead(" + inner + ", k=" + k + ", alpha=" + alpha + ")";
    }
}
