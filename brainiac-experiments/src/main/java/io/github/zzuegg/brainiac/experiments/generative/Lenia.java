package io.github.zzuegg.brainiac.experiments.generative;


import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Lenia (Chan, <a href="https://arxiv.org/abs/1812.05433">1812.05433</a>): a cellular automaton
 * with continuous states, space and time, in which hundreds of life-like "species" glide,
 * rotate and divide. Each step every cell measures its neighbourhood with a smooth ring
 * kernel {@code K} (a bump of radius {@code R}, normalized to sum 1) and moves its state by
 * a growth function of that measure:
 * {@code A <- clip(A + dt * G(K * A), 0, 1)}, {@code G(u) = 2 exp(-(u - mu)^2 / 2 sigma^2) - 1}.
 * The world wraps around at the edges.
 *
 * <p>Besides the simulator there is a tiny {@link #search parameter search} for growth
 * settings under which random soups neither die out nor fill the world, a starting point
 * for evolving species, and a GIF renderer.
 */
public final class Lenia {
    private final int size;
    private final int radius;
    private final double mu;
    private final double sigma;
    private final double dt;
    private final int[] offY;
    private final int[] offX;
    private final double[] weight;
    private double[] state;

    private Lenia(int size, int radius, double mu, double sigma, double dt) {
        this.size = size;
        this.radius = radius;
        this.mu = mu;
        this.sigma = sigma;
        this.dt = dt;
        this.state = new double[size * size];
        List<double[]> taps = new ArrayList<>();
        double sum = 0;
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                double r = Math.hypot(dy, dx) / radius;
                if (r > 0 && r < 1) {
                    double k = Math.exp(4 - 1 / (r * (1 - r)));   // the smooth "bump" shell
                    taps.add(new double[] {dy, dx, k});
                    sum += k;
                }
            }
        }
        offY = new int[taps.size()];
        offX = new int[taps.size()];
        weight = new double[taps.size()];
        for (int i = 0; i < taps.size(); i++) {
            offY[i] = (int) taps.get(i)[0];
            offX[i] = (int) taps.get(i)[1];
            weight[i] = taps.get(i)[2] / sum;
        }
    }

    /**
     * Creates an empty world.
     *
     * @param size the side of the square, wrapping world
     * @param radius the kernel radius R in cells
     * @param mu the growth centre
     * @param sigma the growth width
     * @param dt the time step
     * @return the world
     */
    public static Lenia of(int size, int radius, double mu, double sigma, double dt) {
        return new Lenia(size, radius, mu, sigma, dt);
    }

    /**
     * The classic Orbium settings: R = 13, mu = 0.15, sigma = 0.015, dt = 0.1.
     *
     * @param size the world side
     * @return the world
     */
    public static Lenia orbium(int size) {
        return of(size, 13, 0.15, 0.015, 0.1);
    }

    /** @return the kernel weights (they sum to 1) */
    public double[] kernel() {
        return weight.clone();
    }

    /**
     * The growth function.
     *
     * @param u the neighbourhood measure
     * @return the growth in [-1, 1]
     */
    public double growth(double u) {
        double d = (u - mu) / sigma;
        return 2 * Math.exp(-d * d / 2) - 1;
    }

    /** @return a copy of the state, row-major */
    public double[] state() {
        return state.clone();
    }

    /**
     * Replaces the state.
     *
     * @param values {@code size * size} values in [0, 1]
     */
    public void state(double[] values) {
        if (values.length != size * size) {
            throw new IllegalArgumentException("need " + size * size + " values");
        }
        state = values.clone();
    }

    /**
     * Fills a centred square with random values, a "soup".
     *
     * @param side the side of the square
     * @param random the random source
     */
    public void soup(int side, RandomGenerator random) {
        state = new double[size * size];
        int start = (size - side) / 2;
        for (int y = start; y < start + side; y++) {
            for (int x = start; x < start + side; x++) {
                state[y * size + x] = random.nextDouble();
            }
        }
    }

    /** @return the total mass, the sum of all states */
    public double mass() {
        double m = 0;
        for (double v : state) {
            m += v;
        }
        return m;
    }

    /**
     * The neighbourhood measure {@code K * A} at every cell.
     *
     * @return {@code size * size} values
     */
    public double[] potential() {
        double[] u = new double[size * size];
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                double s = 0;
                for (int i = 0; i < weight.length; i++) {
                    int yy = Math.floorMod(y + offY[i], size);
                    int xx = Math.floorMod(x + offX[i], size);
                    s += weight[i] * state[yy * size + xx];
                }
                u[y * size + x] = s;
            }
        }
        return u;
    }

    /** Advances one time step. */
    public void step() {
        double[] u = potential();
        for (int i = 0; i < state.length; i++) {
            state[i] = Math.max(0, Math.min(1, state[i] + dt * growth(u[i])));
        }
    }

    /**
     * Advances several steps.
     *
     * @param steps how many
     * @param frames receives a float copy of the state every {@code every} steps, or {@code null}
     * @param every the frame interval
     */
    public void run(int steps, List<float[]> frames, int every) {
        for (int s = 0; s < steps; s++) {
            step();
            if (frames != null && s % every == 0) {
                float[] f = new float[state.length];
                for (int i = 0; i < f.length; i++) {
                    f[i] = (float) state[i];
                }
                frames.add(f);
            }
        }
    }

    /**
     * Renders recorded frames as an animated GIF.
     *
     * @param frames recorded frames
     * @param scale the pixel upscaling factor
     * @return the GIF file's bytes
     */
    public byte[] gif(List<float[]> frames, int scale) {
        return Nets.grayGif(frames, size, size, 0, 1, scale, 5);
    }

    /**
     * Searches growth settings for "life": random soups that, after a while, still hold a
     * mass between 0.5% and 40% of the world.
     *
     * @param size the world side
     * @param radius the kernel radius
     * @param trials how many random settings to try
     * @param steps how long each runs
     * @param random the random source
     * @return {@code {mu, sigma}} pairs that survived
     */
    public static List<double[]> search(int size, int radius, int trials, int steps, RandomGenerator random) {
        List<double[]> alive = new ArrayList<>();
        for (int t = 0; t < trials; t++) {
            double mu = 0.1 + 0.25 * random.nextDouble();
            double sigma = 0.01 + 0.05 * random.nextDouble();
            Lenia world = of(size, radius, mu, sigma, 0.1);
            world.soup(size / 2, random);
            world.run(steps, null, 1);
            double share = world.mass() / (size * size);
            if (share > 0.005 && share < 0.4) {
                alive.add(new double[] {mu, sigma});
            }
        }
        return alive;
    }
}
