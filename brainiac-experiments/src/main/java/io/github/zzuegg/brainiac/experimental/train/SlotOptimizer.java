package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Base of the experimental optimizers that keep a few tensors ("slots") per parameter,
 * the public twin of the core's package-private {@code StatefulOptimizer}: a subclass
 * writes one parameter's update, and this class runs the step, counts steps, clears the
 * gradients and saves and restores the slots for checkpoints in the core's key format
 * ({@code "<index>.<slot>"} plus {@code "step"}). Optimizers with global state (such as
 * Prodigy's distance estimate) add it through {@link #extraState} and
 * {@link #loadExtraState}.
 */
public abstract class SlotOptimizer implements Optimizer {

    private float learningRate;
    private final String[] slots;
    private final Map<Var, Tensor[]> state = new IdentityHashMap<>();
    private int steps;

    /**
     * @param learningRate the step size, positive
     * @param slots the names of the tensors kept per parameter
     */
    protected SlotOptimizer(float learningRate, String... slots) {
        this.learningRate = checkedRate(learningRate);
        this.slots = slots.clone();
    }

    /**
     * Updates one parameter: assigns its new value and returns its new slots.
     *
     * @param p the parameter
     * @param g its gradient
     * @param s its slots, {@code null} entries before the first step
     * @param t the step number, from 1
     * @return the new slots
     */
    protected abstract Tensor[] update(Var p, Tensor g, Tensor[] s, int t);

    /**
     * Called once per step before any parameter is updated, for optimizers that need a
     * global view of all gradients (such as a global norm). Does nothing by default.
     *
     * @param withGrad the parameters that have a gradient this step
     * @param t the step number, from 1
     */
    protected void beforeStep(List<Var> withGrad, int t) {
    }

    /**
     * Called once per step after every parameter is updated. Does nothing by default.
     *
     * @param withGrad the parameters that had a gradient
     * @param t the step number
     */
    protected void afterStep(List<Var> withGrad, int t) {
    }

    @Override
    public void step(List<Var> parameters) {
        steps++;
        List<Var> withGrad = new ArrayList<>();
        for (Var p : parameters) {
            if (p.grad() != null) {
                withGrad.add(p);
            }
        }
        beforeStep(withGrad, steps);
        List<Tensor> touched = new ArrayList<>();
        for (Var p : withGrad) {
            Tensor[] old = state.computeIfAbsent(p, k -> new Tensor[slots.length]);
            Tensor[] now = update(p, p.grad(), old.clone(), steps);
            state.put(p, now);
            p.zeroGrad();
            touched.add(p.value());
            for (Tensor x : now) {
                if (x != null) {
                    touched.add(x);
                }
            }
        }
        Tensor.evaluate(touched);
        afterStep(withGrad, steps);
    }

    /** Drops the slots kept for a parameter, so it starts fresh if it is trained again. */
    @Override
    public void forget(Var parameter) {
        state.remove(parameter);
    }

    /** @return the steps taken */
    protected int steps() {
        return steps;
    }

    /**
     * @param p a parameter
     * @return its slots, or {@code null} before its first update
     */
    protected Tensor[] slots(Var p) {
        return state.get(p);
    }

    /**
     * Global state to save besides the slots and the step count.
     *
     * @return named scalar or tensor state; empty by default
     */
    protected Map<String, Tensor> extraState() {
        return Map.of();
    }

    /**
     * Restores what {@link #extraState} saved.
     *
     * @param saved the whole saved map
     */
    protected void loadExtraState(Map<String, Tensor> saved) {
    }

    @Override
    public Map<String, Tensor> state(List<Var> parameters) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        out.put("step", Tensor.of(Shape.of(1), steps));
        out.putAll(extraState());
        for (int i = 0; i < parameters.size(); i++) {
            Tensor[] s = state.get(parameters.get(i));
            if (s == null) {
                continue;
            }
            for (int k = 0; k < slots.length; k++) {
                if (s[k] != null) {
                    out.put(i + "." + slots[k], s[k]);
                }
            }
        }
        return out;
    }

    @Override
    public void loadState(List<Var> parameters, Map<String, Tensor> saved) {
        state.clear();
        Tensor step = saved.get("step");
        steps = step == null ? 0 : Math.round(step.item());
        loadExtraState(saved);
        for (int i = 0; i < parameters.size(); i++) {
            Tensor[] s = new Tensor[slots.length];
            boolean any = false;
            for (int k = 0; k < slots.length; k++) {
                Tensor x = saved.get(i + "." + slots[k]);
                if (x != null) {
                    s[k] = x;
                    any = true;
                }
            }
            if (any) {
                state.put(parameters.get(i), s);
            }
        }
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float learningRate) {
        this.learningRate = checkedRate(learningRate);
    }

    private static float checkedRate(float learningRate) {
        if (!(learningRate > 0)) {
            throw new IllegalArgumentException("learningRate must be positive, got " + learningRate);
        }
        return learningRate;
    }
}
