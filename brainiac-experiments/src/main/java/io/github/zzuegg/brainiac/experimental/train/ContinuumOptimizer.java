package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * An optimizer that updates groups of parameters at different frequencies, the "continuum
 * memory system" of Nested Learning (Behrouz et al. 2025, "Nested Learning: The Illusion of
 * Deep Learning Architectures", <a href="https://arxiv.org/abs/2512.24695">arXiv:2512.24695</a>).
 * A group with period {@code C} accumulates its gradients for {@code C} steps and then applies
 * their mean with its own inner optimizer, so fast groups track the recent data while slow
 * groups integrate over long spans and change rarely: a spectrum of memories from working
 * memory to long-term knowledge, all trained by the same loss.
 *
 * <p>Parameters not in any group are updated every step. Works with any core
 * {@link Optimizer} as the inner one; each group gets its own instance.
 *
 * <pre>{@code
 * Optimizer opt = ContinuumOptimizer.builder(() -> Optimizer.adam(1e-3f))
 *         .group(fastParams, 1).group(slowParams, 16).build();
 * }</pre>
 */
public final class ContinuumOptimizer implements Optimizer {

    private record Group(List<Var> params, int period, Optimizer inner, Map<Var, Tensor> sums) {
    }

    private final List<Group> groups;
    private final Map<Var, Group> owner = new IdentityHashMap<>();
    private final Optimizer rest;
    private long steps;
    private float learningRate;

    private ContinuumOptimizer(List<Group> groups, Optimizer rest) {
        this.groups = groups;
        this.rest = rest;
        this.learningRate = rest.learningRate();
        for (Group g : groups) {
            for (Var p : g.params) {
                owner.put(p, g);
            }
        }
    }

    /**
     * Starts building.
     *
     * @param inner makes the inner optimizer of each group (and of ungrouped parameters)
     * @return the builder
     */
    public static Builder builder(Supplier<Optimizer> inner) {
        return new Builder(inner);
    }

    /** Collects the groups. */
    public static final class Builder {
        private final Supplier<Optimizer> inner;
        private final List<Group> groups = new ArrayList<>();

        private Builder(Supplier<Optimizer> inner) {
            this.inner = inner;
        }

        /**
         * Adds a group.
         *
         * @param params its parameters
         * @param period update every this many steps, at least 1
         * @return this builder
         */
        public Builder group(List<Var> params, int period) {
            if (period < 1) {
                throw new IllegalArgumentException("period must be at least 1");
            }
            groups.add(new Group(List.copyOf(params), period, inner.get(), new IdentityHashMap<>()));
            return this;
        }

        /** @return the optimizer */
        public ContinuumOptimizer build() {
            return new ContinuumOptimizer(List.copyOf(groups), inner.get());
        }
    }

    /** @return how many steps were taken */
    public long steps() {
        return steps;
    }

    @Override
    public void step(List<Var> parameters) {
        steps++;
        List<Var> ungrouped = new ArrayList<>();
        for (Var p : parameters) {
            Group g = owner.get(p);
            if (g == null) {
                ungrouped.add(p);
            } else if (p.grad() != null) {
                g.sums.merge(p, p.grad(), Tensor::add);
                p.zeroGrad();
            }
        }
        if (!ungrouped.isEmpty()) {
            rest.step(ungrouped);
        }
        for (Group g : groups) {
            if (steps % g.period != 0 || g.sums.isEmpty()) {
                continue;
            }
            List<Var> due = new ArrayList<>();
            for (Var p : g.params) {
                Tensor sum = g.sums.remove(p);
                if (sum != null) {
                    p.setGrad(sum.div(g.period));
                    due.add(p);
                }
            }
            g.inner.step(due);
        }
    }

    @Override
    public void forget(Var parameter) {
        Group g = owner.get(parameter);
        if (g == null) {
            rest.forget(parameter);
        } else {
            g.sums.remove(parameter);
            g.inner.forget(parameter);
        }
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    /** Scales every group's learning rate to this one, keeping their ratios. */
    @Override
    public void learningRate(float learningRate) {
        float ratio = learningRate / this.learningRate;
        this.learningRate = learningRate;
        rest.learningRate(rest.learningRate() * ratio);
        for (Group g : groups) {
            g.inner.learningRate(g.inner.learningRate() * ratio);
        }
    }

    @Override
    public String toString() {
        StringBuilder s = new StringBuilder("ContinuumOptimizer(");
        for (Group g : groups) {
            s.append("every ").append(g.period).append(": ").append(g.params.size()).append(" params; ");
        }
        return s.append(")").toString();
    }
}
