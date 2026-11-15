package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;

/**
 * Moves values between channels and space: {@code [c * r^d, s...]} to
 * {@code [c, s * r...]} (shuffle, sub-pixel upsampling) or back (unshuffle).
 */
public final class PixelShuffle implements Layer {

    private final int factor;
    private final boolean inverse;

    /**
     * Creates a pixel shuffle layer.
     *
     * @param factor the scale factor per spatial dimension
     * @param inverse {@code true} for unshuffle (space to channels)
     */
    public PixelShuffle(int factor, boolean inverse) {
        if (factor < 1) {
            throw new IllegalArgumentException("factor must be at least 1, got " + factor);
        }
        this.factor = factor;
        this.inverse = inverse;
    }

    @Override
    public Var forward(Var input) {
        return inverse ? Functional.pixelUnshuffle(input, factor) : Functional.pixelShuffle(input, factor);
    }

    @Override
    public Shape outputShape(Shape input) {
        int r = input.rank() - 1;
        int scale = (int) Math.pow(factor, r);
        int[] out = input.toArray();
        if (inverse) {
            out[0] *= scale;
            for (int i = 1; i <= r; i++) {
                if (out[i] % factor != 0) {
                    throw new IllegalArgumentException(this + " needs spatial sizes divisible by " + factor + ", got "
                            + input);
                }
                out[i] /= factor;
            }
        } else {
            if (out[0] % scale != 0) {
                throw new IllegalArgumentException(this + " needs channels divisible by " + scale + ", got " + input);
            }
            out[0] /= scale;
            for (int i = 1; i <= r; i++) {
                out[i] *= factor;
            }
        }
        return Shape.of(out);
    }

    @Override
    public String toString() {
        return (inverse ? "PixelUnshuffle(" : "PixelShuffle(") + factor + ")";
    }

    static Config write(PixelShuffle p) {
        return Config.builder().put("factor", p.factor).put("inverse", p.inverse).build();
    }

    static PixelShuffle read(Config c) {
        return new PixelShuffle(c.getInt("factor"), c.getBoolean("inverse"));
    }
}
