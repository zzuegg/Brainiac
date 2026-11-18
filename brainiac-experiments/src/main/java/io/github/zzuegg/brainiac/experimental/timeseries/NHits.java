package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * N-HiTS (Challu et al., AAAI 2023, <a href="https://arxiv.org/abs/2201.12886">arXiv:2201.12886</a>),
 * N-BEATS with hierarchical interpolation. The doubly residual stack of N-BEATS
 * ({@code experimental.nn.NBeats}: each block subtracts its backcast from the input and adds
 * its forecast to the output) gets two changes:
 * <ul>
 *   <li><b>multi-rate sampling</b>: each stack max-pools its input with its own kernel
 *       before the block's MLP, so early stacks see a coarse, low-frequency view;</li>
 *   <li><b>hierarchical interpolation</b>: each block predicts only
 *       {@code ceil(horizon / ratio)} forecast coefficients, which a fixed linear
 *       interpolation matrix spreads over the horizon; early stacks use large ratios (few
 *       coefficients, smooth forecasts), the last stack ratio 1.</li>
 * </ul>
 * The result needs far fewer parameters than N-BEATS at long horizons and specialises stacks
 * by frequency band. Input {@code [batch, lookback]}, output {@code [batch, horizon]};
 * instance normalisation of the window (subtract its mean, divide by its spread) is built in.
 */
public final class NHits implements Layer {

    private final int lookback;
    private final int horizon;
    private final List<Block> blocks = new ArrayList<>();

    private static final class Block {
        final int pool;
        final List<Var> fc = new ArrayList<>();
        final Var back;
        final Var fore;
        final Tensor interpolation;

        Block(int lookback, int horizon, int pool, int ratio, int hidden, int layers, SplittableRandom r) {
            this.pool = pool;
            int in = (lookback + pool - 1) / pool;
            for (int l = 0; l < layers; l++) {
                fc.add(Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) Math.sqrt(2.0 / in), r)));
                fc.add(Var.param(Tensor.zeros(Shape.of(1, hidden))));
                in = hidden;
            }
            int coefficients = Math.max(1, (horizon + ratio - 1) / ratio);
            back = Var.param(Tensor.normal(Shape.of(hidden, lookback), 0, (float) Math.sqrt(0.1 / hidden), r));
            fore = Var.param(Tensor.normal(Shape.of(hidden, coefficients), 0, (float) Math.sqrt(0.1 / hidden), r));
            interpolation = interpolationMatrix(coefficients, horizon);
        }

        Var[] apply(Var x, int lookback) {
            int b = x.shape().dim(0);
            Var h = x;
            if (pool > 1) {
                int padded = ((lookback + pool - 1) / pool) * pool;
                if (padded > lookback) {
                    h = Var.concat(1, x.slice(1, 0, padded - lookback), x);
                }
                h = h.reshape(b, padded / pool, pool).max(2, false);
            }
            for (int l = 0; l < fc.size(); l += 2) {
                h = h.matmul(fc.get(l)).add(fc.get(l + 1)).relu();
            }
            return new Var[] {h.matmul(back), h.matmul(fore).matmul(Var.of(interpolation))};
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(fc);
            p.add(back);
            p.add(fore);
            return p;
        }
    }

    /**
     * The linear interpolation from {@code coefficients} knots, evenly spread over the
     * horizon, to every step.
     *
     * @param coefficients the knots
     * @param horizon the steps
     * @return {@code [coefficients, horizon]}
     */
    public static Tensor interpolationMatrix(int coefficients, int horizon) {
        float[] m = new float[coefficients * horizon];
        for (int t = 0; t < horizon; t++) {
            if (coefficients == 1) {
                m[t] = 1;
                continue;
            }
            double pos = t * (coefficients - 1) / (double) Math.max(1, horizon - 1);
            int lo = (int) Math.floor(pos);
            int hi = Math.min(coefficients - 1, lo + 1);
            double w = pos - lo;
            m[lo * horizon + t] += (float) (1 - w);
            m[hi * horizon + t] += (float) w;
        }
        return Tensor.of(Shape.of(coefficients, horizon), m);
    }

    /**
     * Creates the model.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param pools the max-pool kernel of each stack, for example {8, 4, 1}
     * @param ratios the expressiveness ratio of each stack, for example {24, 6, 1}
     * @param blocksPerStack blocks per stack
     * @param hidden MLP width
     * @param layers MLP depth
     * @param random the initialisation source
     */
    public NHits(int lookback, int horizon, int[] pools, int[] ratios, int blocksPerStack, int hidden, int layers,
            SplittableRandom random) {
        this.lookback = lookback;
        this.horizon = horizon;
        for (int s = 0; s < pools.length; s++) {
            for (int k = 0; k < blocksPerStack; k++) {
                blocks.add(new Block(lookback, horizon, pools[s], ratios[s], hidden, layers, random));
            }
        }
    }

    /**
     * The paper's default three-stack layout: pools {@code {8, 4, 1}} (capped by the lookback)
     * and ratios {@code {horizon / 4, horizon / 12, 1}}-style, one block per stack.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param hidden MLP width
     * @param seed the seed
     * @return the model
     */
    public static NHits standard(int lookback, int horizon, int hidden, long seed) {
        int r1 = Math.max(1, horizon / 4);
        int r2 = Math.max(1, horizon / 12);
        return new NHits(lookback, horizon, new int[] {Math.min(8, lookback / 4), Math.min(4, lookback / 8 + 1), 1},
                new int[] {r1, r2, 1}, 1, hidden, 2, new SplittableRandom(seed));
    }

    /**
     * @param horizon forecast steps
     * @param hidden MLP width
     * @return a spec with the standard layout
     */
    public static LayerSpec spec(int horizon, int hidden) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(horizon);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return standard(input.dim(-1), horizon, hidden, random.nextLong());
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var mean = input.mean(1, true);
        Var std = input.std(1, 0, true).add(1e-4f);
        Var residual = input.sub(mean).div(std);
        Var forecast = null;
        for (Block b : blocks) {
            Var[] bf = b.apply(residual, lookback);
            residual = residual.sub(bf[0]);
            forecast = forecast == null ? bf[1] : forecast.add(bf[1]);
        }
        return forecast.mul(std).add(mean);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Block b : blocks) {
            p.addAll(b.parameters());
        }
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return horizon;
    }

    /** @return the number of trainable scalars */
    public long parameterCount() {
        long n = 0;
        for (Var v : parameters()) {
            n += v.shape().size();
        }
        return n;
    }

    @Override
    public String toString() {
        return "NHits(" + lookback + " -> " + horizon + ", blocks=" + blocks.size() + ")";
    }
}
