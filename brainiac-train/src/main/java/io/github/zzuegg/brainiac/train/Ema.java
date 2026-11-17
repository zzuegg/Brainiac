package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * An exponential moving average of a model's weights, which often evaluates better than the
 * weights themselves. Call {@link #update} after each optimizer step and {@link #swap} to
 * evaluate with the averages (and again to go back).
 *
 * <pre>{@code
 * Ema ema = new Ema(model.parameters(), 0.999f);
 * for (...) { loss.backward(); optimizer.step(params); ema.update(); }
 * ema.swap();   // evaluate or save the averaged model
 * ema.swap();   // and continue training
 * }</pre>
 */
public final class Ema {

    private final List<Var> parameters;
    private final float decay;
    private final boolean warmup;
    private final List<Tensor> averages = new ArrayList<>();
    private int updates;

    /**
     * Starts the averages at the current weights.
     *
     * @param parameters the weights to average
     * @param decay how much of the average each update keeps, such as {@code 0.999f}
     */
    public Ema(List<Var> parameters, float decay) {
        this(parameters, decay, false);
    }

    /**
     * Starts the averages at the current weights.
     *
     * @param parameters the weights to average
     * @param decay how much of the average each update keeps
     * @param warmup whether to keep less early on: {@code min(decay, (1 + n) / (10 + n))}
     *     after {@code n} updates, so the average forgets the random start sooner
     */
    public Ema(List<Var> parameters, float decay, boolean warmup) {
        if (decay < 0 || decay >= 1) {
            throw new IllegalArgumentException("decay must be in [0, 1), got " + decay);
        }
        this.parameters = List.copyOf(parameters);
        this.decay = decay;
        this.warmup = warmup;
        for (Var p : this.parameters) {
            averages.add(p.value().mul(1));
        }
        Tensor.evaluate(averages);
    }

    /** @return the decay used by the next update */
    public float currentDecay() {
        return warmup ? Math.min(decay, (1f + updates) / (10f + updates)) : decay;
    }

    /** Moves the averages towards the current weights. */
    public void update() {
        float d = currentDecay();
        updates++;
        List<Tensor> old = new ArrayList<>(averages);
        List<Tensor> created = Tensor.created(() -> {
            for (int i = 0; i < parameters.size(); i++) {
                averages.set(i, averages.get(i).mul(d).add(parameters.get(i).value().mul(1 - d)));
            }
        });
        Tensor.evaluate(averages);
        old.addAll(created);
        Tensor.release(old, averages);
    }

    /** Exchanges the weights and the averages: call once to use the averages, again to go back. */
    public void swap() {
        for (int i = 0; i < parameters.size(); i++) {
            Tensor live = parameters.get(i).value();
            parameters.get(i).assign(averages.get(i));
            averages.set(i, live);
        }
    }

    /** @return the updates so far */
    public int updates() {
        return updates;
    }

    /**
     * The averages, for saving.
     *
     * @return {@code "<index>"} to the average of parameter {@code index}, and {@code "updates"}
     */
    public Map<String, Tensor> state() {
        Map<String, Tensor> out = new LinkedHashMap<>();
        out.put("updates", Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(1), updates));
        for (int i = 0; i < averages.size(); i++) {
            out.put(Integer.toString(i), averages.get(i));
        }
        return out;
    }

    /**
     * Restores saved averages.
     *
     * @param state what {@link #state} returned
     */
    public void loadState(Map<String, Tensor> state) {
        updates = Math.round(state.get("updates").item());
        for (int i = 0; i < averages.size(); i++) {
            Tensor t = state.get(Integer.toString(i));
            averages.set(i, t.to(parameters.get(i).value().device()));
        }
    }
}
