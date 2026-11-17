package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Reversible instance normalization (Kim et al., 2022,
 * <a href="https://openreview.net/forum?id=cGDAkQo1C0p">ICLR 2022</a>) for forecasting: each input
 * window is standardized by its own mean and standard deviation before the model, and the
 * model's forecast is mapped back with the same statistics. It removes level and scale
 * shifts between windows (distribution shift over time), so the model only has to learn
 * shapes; nearly every modern forecaster (PatchTST, DLinear variants, TimesNet) uses it.
 */
public final class RevIn {

    private RevIn() {
    }

    /**
     * Per-row statistics of a batch of windows.
     *
     * @param mean {@code [batch, 1]}
     * @param std {@code [batch, 1]}
     */
    public record Stats(Tensor mean, Tensor std) {
    }

    /**
     * @param windows {@code [batch, steps]}
     * @return their means and standard deviations (constants: no gradient flows through them)
     */
    public static Stats stats(Tensor windows) {
        Tensor mean = windows.mean(1, true);
        Tensor std = windows.sub(mean).square().mean(1, true).add(1e-5f).sqrt();
        return new Stats(mean, std);
    }

    /**
     * @param x {@code [batch, steps]}
     * @param s statistics
     * @return {@code (x - mean) / std}
     */
    public static Var normalize(Var x, Stats s) {
        return x.sub(s.mean()).div(s.std());
    }

    /**
     * @param y {@code [batch, horizon]} in normalized units
     * @param s the statistics of the input windows
     * @return {@code y * std + mean}
     */
    public static Var denormalize(Var y, Stats s) {
        return y.mul(s.std()).add(s.mean());
    }
}
