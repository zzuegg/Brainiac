package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Stochastic rounding to bfloat16 for training with low-precision weights and no float32
 * master copy (Gupta et al. 2015, <a href="https://arxiv.org/abs/1502.02551">arXiv:1502.02551</a>;
 * Zamirai et al. 2020, "Revisiting BFloat16 Training",
 * <a href="https://arxiv.org/abs/2010.06192">arXiv:2010.06192</a>). bfloat16 keeps 8
 * mantissa bits, so near a weight of 1 the representable values are {@code 2^-7 ≈ 0.008}
 * apart: an update smaller than half of that is lost entirely by round-to-nearest, and
 * pure-bf16 training stalls once updates shrink. Rounding up with probability equal to the
 * remainder's fraction of the gap makes the rounding unbiased ({@code E[round(x)] = x}), so
 * small updates still move the weights on average.
 *
 * <p>{@link #bf16Weights} wraps any optimizer so that after each step every parameter is
 * put back on the bfloat16 grid, stochastically or to nearest; the inner optimizer's state
 * stays float32 (as in bf16 training with float32 optimizer state). Values are kept as
 * float32 tensors holding bfloat16-representable numbers, so the rest of the model is
 * unchanged.
 */
public final class StochasticRounding {

    private StochasticRounding() {
    }

    /**
     * Rounds to the nearest bfloat16 value (ties to even), through the core conversion.
     *
     * @param x the values
     * @return float32 values on the bfloat16 grid
     */
    public static Tensor nearestBf16(Tensor x) {
        return x.to(DType.BFLOAT16).to(DType.FLOAT32);
    }

    /**
     * Rounds to one of the two neighbouring bfloat16 values, the upper one (in magnitude)
     * with probability equal to the remainder's share of the gap: adds 16 random bits below
     * the bfloat16 mantissa and truncates.
     *
     * @param x the values
     * @param random the random source
     * @return float32 values on the bfloat16 grid
     */
    public static Tensor stochasticBf16(Tensor x, RandomGenerator random) {
        float[] v = x.toArray();
        for (int i = 0; i < v.length; i++) {
            v[i] = stochasticBf16(v[i], random.nextInt(1 << 16));
        }
        return Tensor.of(x.shape(), v, x.device());
    }

    /**
     * One value: adds {@code noise} to the 16 bits bfloat16 drops, then drops them.
     *
     * @param x the value
     * @param noise a random number in {@code [0, 65536)}
     * @return the rounded value
     */
    public static float stochasticBf16(float x, int noise) {
        if (!Float.isFinite(x)) {
            return x;
        }
        int bits = Float.floatToRawIntBits(x);
        int rounded = (bits + noise) & 0xFFFF0000;
        float r = Float.intBitsToFloat(rounded);
        return Float.isFinite(r) ? r : Float.intBitsToFloat(bits & 0xFFFF0000);
    }

    /**
     * An optimizer whose weights live on the bfloat16 grid.
     *
     * @param inner the optimizer computing the float32 update
     * @param stochastic whether to round stochastically (else to nearest)
     * @param seed the seed of the rounding noise
     * @return the wrapped optimizer
     */
    public static Optimizer bf16Weights(Optimizer inner, boolean stochastic, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        return new Optimizer() {
            @Override
            public void step(List<Var> parameters) {
                List<Var> live = parameters.stream().filter(p -> p.grad() != null).toList();
                inner.step(parameters);
                for (Var p : live) {
                    p.assign(stochastic ? stochasticBf16(p.value(), random) : nearestBf16(p.value()));
                }
            }

            @Override
            public void resetState(Var parameter, Tensor keep) {
                inner.resetState(parameter, keep);
            }

            @Override
            public void forget(Var parameter) {
                inner.forget(parameter);
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
                return inner.state(parameters);
            }

            @Override
            public void loadState(List<Var> parameters, Map<String, Tensor> state) {
                inner.loadState(parameters, state);
            }

            @Override
            public String toString() {
                return inner + " on a bfloat16 grid (" + (stochastic ? "stochastic" : "nearest") + " rounding)";
            }
        };
    }
}
