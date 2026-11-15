package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Adds position information to sequences {@code [steps, features]}: fixed sinusoids or a
 * learned vector per position (for at most {@code maxSteps} steps).
 */
public final class PositionalEncoding implements Layer {

    private final int features;
    private final int maxSteps;
    private final float base;
    private final Var table;

    private PositionalEncoding(int features, int maxSteps, float base, Var table) {
        if (features < 1 || maxSteps < 1) {
            throw new IllegalArgumentException("features and maxSteps must be at least 1");
        }
        this.features = features;
        this.maxSteps = maxSteps;
        this.base = base;
        this.table = table;
    }

    /**
     * Fixed sinusoidal encodings.
     *
     * @param features the features per step
     * @return the layer
     */
    public static PositionalEncoding sinusoidal(int features) {
        return new PositionalEncoding(features, Integer.MAX_VALUE, 10000, null);
    }

    /**
     * Learned encodings, one vector per position, starting small.
     *
     * @param maxSteps the longest sequence
     * @param features the features per step
     * @param random the random source
     * @return the layer
     */
    public static PositionalEncoding learned(int maxSteps, int features, SplittableRandom random) {
        return new PositionalEncoding(features, maxSteps, 0,
                Var.param(Tensor.normal(Shape.of(maxSteps, features), 0, 0.02f, random)));
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != features || s.dim(1) > maxSteps) {
            throw new IllegalArgumentException(this + " expects [batch, steps <= " + maxSteps + ", " + features
                    + "], got " + s);
        }
        int steps = s.dim(1);
        if (table == null) {
            return input.add(Functional.sinusoidal(steps, features, base, input.value().device()));
        }
        return input.add(steps == maxSteps ? table : table.slice(0, 0, steps));
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public List<Var> parameters() {
        return table == null ? List.of() : List.of(table);
    }

    @Override
    public String toString() {
        return table == null ? "SinusoidalPositions" : "LearnedPositions(" + maxSteps + ")";
    }

    static Config write(PositionalEncoding p) {
        return Config.builder().put("features", p.features).put("maxSteps", p.maxSteps).put("base", p.base)
                .put("learned", p.table != null).build();
    }

    static PositionalEncoding read(Config c) {
        int f = c.getInt("features");
        return c.getBoolean("learned")
                ? new PositionalEncoding(f, c.getInt("maxSteps"), 0, Var.param(Tensor.zeros(Shape.of(c.getInt("maxSteps"), f))))
                : new PositionalEncoding(f, c.getInt("maxSteps"), c.getFloat("base"), null);
    }
}
