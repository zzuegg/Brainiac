package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * An optimizer that keeps a few tensors per parameter, named slots (such as Adam's
 * {@code "m"} and {@code "v"}). Subclasses write one parameter's update; this class runs the
 * step as one batch of tensor work, frees what the step no longer needs, saves and restores
 * the state, and resets it for chosen entries.
 */
public abstract class StatefulOptimizer implements Optimizer {

    private float learningRate;
    private final String[] slots;
    private final Map<Var, Tensor[]> state = new IdentityHashMap<>();
    private int steps;

    /**
     * @param learningRate the step size
     * @param slots the names of the tensors kept per parameter
     */
    protected StatefulOptimizer(float learningRate, String... slots) {
        this.learningRate = positive(learningRate);
        this.slots = slots.clone();
    }

    /**
     * Updates one parameter: assigns its new value and returns its new state.
     *
     * @param p the parameter
     * @param g its gradient
     * @param s its state, one entry per slot, {@code null} entries before the first step
     * @param t the step number, from 1
     * @return the new state, one entry per slot
     */
    protected abstract Tensor[] update(Var p, Tensor g, Tensor[] s, int t);

    @Override
    public void step(List<Var> parameters) {
        steps++;
        int t = steps;
        List<Tensor> grads = new ArrayList<>(parameters.size());
        for (Var p : parameters) {
            if (p.grad() != null) {
                grads.add(p.grad());
            }
        }
        Tensor.evaluate(grads);   // the gradients' last steps together, rather than one by one
        List<Tensor> updated = new ArrayList<>();
        List<Tensor> previous = new ArrayList<>();
        List<Tensor> created = Tensor.created(() -> {
            for (Var p : parameters) {
                Tensor g = p.grad();
                if (g == null) {
                    continue;
                }
                Tensor[] old = state.computeIfAbsent(p, k -> new Tensor[slots.length]);
                Tensor[] now = update(p, g, old.clone(), t);
                Set<Tensor> kept = Collections.newSetFromMap(new IdentityHashMap<>());
                for (Tensor x : now) {
                    if (x != null) {
                        kept.add(x);
                        updated.add(x);
                    }
                }
                for (Tensor x : old) {
                    if (x != null && !kept.contains(x)) {
                        previous.add(x);
                    }
                }
                state.put(p, now);
                p.zeroGrad();
                updated.add(p.value());
            }
        });
        Tensor.evaluate(updated);   // on a GPU, the whole step as few kernels
        // only this optimizer saw its old state, the gradients and the values on the way
        previous.addAll(created);
        previous.addAll(grads);
        Tensor.release(previous, updated);
    }

    /** @return the steps taken */
    protected int steps() {
        return steps;
    }

    /**
     * Returns one parameter's state.
     *
     * @param parameter the parameter
     * @return its tensors, one per slot ({@code null} before its first step); a copy of the array
     */
    public Tensor[] slots(Var parameter) {
        Tensor[] s = state.get(parameter);
        return s == null ? new Tensor[slots.length] : s.clone();
    }

    /**
     * Resets the state of chosen entries of a parameter, as if they had never been updated,
     * for example when units are re-initialized: each slot of the parameter's shape is
     * multiplied by {@code keep}. Slots of another shape or kept packed (Adafactor's factored
     * ones, Adam8bit's) are reset whole if any entry is.
     *
     * @param parameter the parameter
     * @param keep 1 for entries to keep and 0 for entries to reset, broadcast to the parameter
     */
    @Override
    public void resetState(Var parameter, Tensor keep) {
        Tensor[] s = state.get(parameter);
        if (s == null) {
            return;
        }
        Tensor mask = keep.shape().equals(parameter.shape()) ? keep : keep.expand(parameter.shape());
        boolean any = mask.min().item() < 0.5f;
        for (int k = 0; k < s.length; k++) {
            Tensor x = s[k];
            if (x == null) {
                continue;
            }
            if (!x.isPacked() && x.shape().equals(parameter.shape())) {
                s[k] = x.mul(mask.to(x.dtype()));
            } else if (any) {
                s[k] = null;   // started again from zero at the next step
            }
        }
    }

    @Override
    public Map<String, Tensor> state(List<Var> parameters) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        out.put("step", Tensor.of(Shape.of(1), steps));
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
        for (int i = 0; i < parameters.size(); i++) {
            Tensor[] s = new Tensor[slots.length];
            boolean any = false;
            for (int k = 0; k < slots.length; k++) {
                Tensor x = saved.get(i + "." + slots[k]);
                if (x != null) {
                    Var p = parameters.get(i);
                    s[k] = x.device().equals(p.value().device()) ? x : x.to(p.value().device());
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
        this.learningRate = positive(learningRate);
    }

    private static float positive(float learningRate) {
        if (!(learningRate > 0)) {
            throw new IllegalArgumentException("learningRate must be positive, got " + learningRate);
        }
        return learningRate;
    }
}
