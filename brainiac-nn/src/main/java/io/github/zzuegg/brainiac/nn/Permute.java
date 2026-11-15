package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Arrays;

/**
 * Reorders the dimensions of each sample, for example {@code new Permute(1, 0)} turns
 * {@code [channels, steps]} into {@code [steps, channels]}.
 */
public final class Permute implements Layer {

    private final int[] axes;

    /**
     * Creates a permute layer.
     *
     * @param axes for each output dimension, the sample dimension it takes (batch excluded)
     */
    public Permute(int... axes) {
        boolean[] seen = new boolean[axes.length];
        for (int a : axes) {
            if (a < 0 || a >= axes.length || seen[a]) {
                throw new IllegalArgumentException("not a permutation: " + Arrays.toString(axes));
            }
            seen[a] = true;
        }
        this.axes = axes.clone();
    }

    @Override
    public Var forward(Var input) {
        int[] full = new int[axes.length + 1];
        for (int i = 0; i < axes.length; i++) {
            full[i + 1] = axes[i] + 1;
        }
        return input.permute(full);
    }

    @Override
    public Shape outputShape(Shape input) {
        if (input.rank() != axes.length) {
            throw new IllegalArgumentException(this + " expects " + axes.length + " sample dimensions, got " + input);
        }
        int[] out = new int[axes.length];
        for (int i = 0; i < axes.length; i++) {
            out[i] = input.dim(axes[i]);
        }
        return Shape.of(out);
    }

    @Override
    public String toString() {
        return "Permute" + Arrays.toString(axes);
    }

    static Config write(Permute p) {
        return Config.builder().put("axes", p.axes).build();
    }

    static Permute read(Config c) {
        return new Permute(c.getInts("axes"));
    }
}
