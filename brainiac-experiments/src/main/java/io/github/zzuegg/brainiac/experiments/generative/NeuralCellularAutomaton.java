package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * X16: Growing Neural Cellular Automata (Mordvintsev et al., <a href="https://distill.pub/2020/growing-ca/">Distill 2020</a>).
 * Every cell of a grid holds a small state vector: channel 0 is the visible value, channel 1
 * the "alive" level, the rest are hidden. One learned rule, the same for every cell, looks
 * at the cell and its 3x3 neighbourhood (identity plus Sobel gradients, a depthwise
 * convolution) and adds a small update through a 1x1-convolution MLP. Updates fire
 * stochastically per cell (no global clock), and only cells with a living neighbour count.
 *
 * <p>Trained by backpropagation through many CA steps to grow a target picture from a single
 * seed cell. A sample pool of earlier results, some of them damaged, teaches the rule to
 * keep the pattern stable and to regrow it after damage: a self-repairing pattern.
 */
public final class NeuralCellularAutomaton {
    private final int channels;
    private final int size;
    private final float[] target;
    private final Var perception;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Optimizer optimizer = Optimizer.adam(6e-3f);
    private final SplittableRandom random;
    private final List<Tensor> pool = new ArrayList<>();

    private NeuralCellularAutomaton(float[] target, int size, int channels, int hidden, long seed) {
        this.target = target.clone();
        this.size = size;
        this.channels = channels;
        this.random = new SplittableRandom(seed);
        float[] k = new float[3 * channels * 9];
        float[] sobel = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        for (int c = 0; c < channels; c++) {
            k[(3 * c) * 9 + 4] = 1;   // identity
            for (int i = 0; i < 9; i++) {
                k[(3 * c + 1) * 9 + i] = sobel[i] / 8f;   // d/dx
                k[(3 * c + 2) * 9 + i] = sobel[(i % 3) * 3 + i / 3] / 8f;   // d/dy
            }
        }
        perception = Var.of(Tensor.of(Shape.of(3 * channels, 1, 3, 3), k));
        w1 = Var.param(Tensor.normal(Shape.of(hidden, 3 * channels, 1, 1), 0, (float) Math.sqrt(2.0 / (3 * channels)),
                random));
        b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
        w2 = Var.param(Tensor.zeros(Shape.of(channels, hidden, 1, 1)));   // "do nothing" at the start
    }

    /**
     * Creates an untrained automaton.
     *
     * @param target the picture to grow, {@code size * size} values in [0, 1]
     * @param size the grid side
     * @param channels the state size per cell, at least 3
     * @param hidden the width of the update MLP
     * @param seed the seed
     * @return the automaton
     */
    public static NeuralCellularAutomaton of(float[] target, int size, int channels, int hidden, long seed) {
        if (target.length != size * size || channels < 3) {
            throw new IllegalArgumentException("target must be size*size and channels at least 3");
        }
        return new NeuralCellularAutomaton(target, size, channels, hidden, seed);
    }

    /**
     * A diamond-shaped target picture: a bright outline around a dimmer filling, centred on
     * the seed cell so growth can start there.
     *
     * @param size the grid side
     * @param radius the diamond's radius in cells
     * @return {@code size * size} values in [0, 1]
     */
    public static float[] diamond(int size, int radius) {
        float[] t = new float[size * size];
        int c = size / 2;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int d = Math.abs(y - c) + Math.abs(x - c);
                t[y * size + x] = d == radius ? 1f : d < radius ? 0.4f : 0f;
            }
        }
        return t;
    }

    /**
     * The starting state: one living cell in the middle.
     *
     * @param batch how many copies
     * @return {@code [batch, channels, size, size]}
     */
    public Tensor seed(int batch) {
        float[] s = new float[batch * channels * size * size];
        int mid = (size / 2) * size + size / 2;
        for (int b = 0; b < batch; b++) {
            for (int c = 1; c < channels; c++) {
                s[(b * channels + c) * size * size + mid] = 1;
            }
        }
        return Tensor.of(Shape.of(batch, channels, size, size), s);
    }

    /** Cells that are alive or next to an alive cell, from the alive channel. */
    private Tensor aliveMask(Tensor state) {
        int b = state.shape().dim(0);
        float[] s = state.toArray();
        float[] m = new float[b * size * size];
        for (int n = 0; n < b; n++) {
            int base = (n * channels + 1) * size * size;
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    float max = 0;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int yy = y + dy;
                            int xx = x + dx;
                            if (yy >= 0 && yy < size && xx >= 0 && xx < size) {
                                max = Math.max(max, s[base + yy * size + xx]);
                            }
                        }
                    }
                    m[n * size * size + y * size + x] = max > 0.1f ? 1 : 0;
                }
            }
        }
        return Tensor.of(Shape.of(b, 1, size, size), m);
    }

    /** One stochastic CA step. */
    Var step(Var state, double fireRate, SplittableRandom r) {
        int b = state.shape().dim(0);
        int[] one = {1, 1};
        int[] zero = {0, 0};
        Tensor before = aliveMask(state.value());
        Var p = Functional.conv(state, perception, null, one, one, one, one, channels);
        Var h = Functional.conv(p, w1, b1, one, zero, zero, one, 1).relu();
        Var dx = Functional.conv(h, w2, null, one, zero, zero, one, 1);
        float[] fire = new float[b * size * size];
        for (int i = 0; i < fire.length; i++) {
            fire[i] = r.nextDouble() < fireRate ? 1 : 0;
        }
        Var next = state.add(dx.mul(Tensor.of(Shape.of(b, 1, size, size), fire)));
        Tensor alive = before.mul(aliveMask(next.value()));
        return next.mul(alive);
    }

    /**
     * Runs the automaton without learning.
     *
     * @param state the start, {@code [batch, channels, size, size]}
     * @param steps how many steps
     * @param seed the seed for the update timing
     * @param frames receives the visible channel of the first state after every step, or {@code null}
     * @return the final state
     */
    public Tensor run(Tensor state, int steps, long seed, List<float[]> frames) {
        SplittableRandom r = new SplittableRandom(seed);
        Var s = Var.of(state);
        for (int i = 0; i < steps; i++) {
            s = Var.of(step(s, 0.5, r).value());
            if (frames != null) {
                frames.add(visible(s.value(), 0));
            }
        }
        return s.value();
    }

    /**
     * Reads the visible channel.
     *
     * @param state a state
     * @param index which batch element
     * @return {@code size * size} values
     */
    public float[] visible(Tensor state, int index) {
        float[] s = state.toArray();
        float[] v = new float[size * size];
        System.arraycopy(s, index * channels * size * size, v, 0, size * size);
        return v;
    }

    /**
     * The mean squared error of the visible and alive channels against the target.
     *
     * @param state a state
     * @return the loss per batch element, averaged
     */
    public float loss(Tensor state) {
        return lossVar(Var.of(state)).value().item();
    }

    private Var lossVar(Var state) {
        int b = state.shape().dim(0);
        float[] t = new float[b * 2 * size * size];
        for (int n = 0; n < b; n++) {
            System.arraycopy(target, 0, t, n * 2 * size * size, size * size);
            for (int i = 0; i < size * size; i++) {
                t[n * 2 * size * size + size * size + i] = target[i] > 0 ? 1 : 0;   // alive where the target is
            }
        }
        return state.slice(1, 0, 2).sub(Var.of(Tensor.of(Shape.of(b, 2, size, size), t))).square().mean();
    }

    /**
     * Zeroes a square of the grid, as damage.
     *
     * @param state a state
     * @param index which batch element
     * @param y0 top row
     * @param x0 left column
     * @param side the square's side
     * @return the damaged copy
     */
    public Tensor damage(Tensor state, int index, int y0, int x0, int side) {
        float[] s = state.toArray();
        for (int c = 0; c < channels; c++) {
            for (int y = y0; y < Math.min(size, y0 + side); y++) {
                for (int x = x0; x < Math.min(size, x0 + side); x++) {
                    s[((index * channels + c) * size + y) * size + x] = 0;
                }
            }
        }
        return Tensor.of(state.shape(), s);
    }

    /**
     * One training iteration: a batch from the sample pool (the worst sample replaced by a
     * fresh seed, some damaged), run a random number of steps, backpropagate through all of
     * them, return the results to the pool.
     *
     * @param batch the batch size
     * @param minSteps the fewest CA steps
     * @param maxSteps the most CA steps
     * @return the loss
     */
    public float trainStep(int batch, int minSteps, int maxSteps) {
        Tensor start;
        if (pool.size() < batch) {
            start = seed(batch);
        } else {
            Tensor[] picks = new Tensor[batch];
            for (int i = 0; i < batch; i++) {
                picks[i] = pool.remove(random.nextInt(pool.size()));
            }
            java.util.Arrays.sort(picks, (a, c) -> Float.compare(loss(c), loss(a)));
            picks[0] = seed(1);
            for (int i = batch - 2; i < batch; i++) {
                if (i > 0) {
                    picks[i] = damage(picks[i], 0, random.nextInt(size - 4), random.nextInt(size - 4), 4);
                }
            }
            start = Tensor.concat(0, picks);
        }
        int steps = minSteps + random.nextInt(maxSteps - minSteps + 1);
        Var s = Var.of(start);
        for (int i = 0; i < steps; i++) {
            s = step(s, 0.5, random);
        }
        Var loss = lossVar(s);
        float value = loss.value().item();
        loss.backward();
        List<Var> params = List.of(w1, b1, w2);
        for (Var p : params) {
            if (p.grad() != null) {   // per-parameter gradient normalization, as in the paper
                p.setGrad(p.grad().div(p.grad().square().sum().sqrt().add(1e-8f)));
            }
        }
        optimizer.step(params);
        Tensor out = s.value();
        for (int i = 0; i < batch; i++) {
            pool.add(out.slice(0, i, i + 1));
        }
        while (pool.size() > 64) {
            pool.remove(random.nextInt(pool.size()));
        }
        return value;
    }

    /**
     * Renders a run as an animated GIF of the visible channel.
     *
     * @param frames what {@link #run} recorded
     * @param scale the pixel upscaling factor
     * @return the GIF file's bytes
     */
    public byte[] gif(List<float[]> frames, int scale) {
        return Nets.grayGif(frames, size, size, 0, 1, scale, 6);
    }
}
