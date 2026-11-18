package io.github.zzuegg.brainiac.experimental.parallel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.IntUnaryOperator;

/**
 * Asynchronous stochastic gradient descent, where workers apply gradients computed on
 * parameters that are several updates old.
 *
 * <p>Two engines:
 * <ul>
 *   <li>{@link #hogwild}: real threads update one shared parameter array without any locks
 *       (Hogwild!). Reads and writes race, but when each gradient touches few coordinates, as in
 *       sparse logistic regression, collisions are rare and the speed-up is nearly linear. A
 *       locked variant serialises the updates for comparison.</li>
 *   <li>{@link #simulate}: a deterministic simulation with injected staleness: update {@code t}
 *       applies a gradient computed at the parameters of update {@code t - tau}, with {@code tau}
 *       drawn from a delay distribution. Options: delay-compensated ASGD (DC-ASGD), which
 *       corrects the stale gradient with a diagonal outer-product approximation of the Hessian,
 *       {@code g + lambda g * g * (w_now - w_old)}, in the adaptive form (DC-ASGD-a) that scales
 *       {@code lambda} by the inverse root mean square of each coordinate's gradient; and staleness-aware learning rates,
 *       {@code lr / max(1, tau)}.</li>
 * </ul>
 * Problems provide sparse gradients through a {@link Problem}, so both engines work with any
 * model written over a flat parameter array.
 *
 * <p>Papers: Niu et al., "Hogwild!", <a href="https://arxiv.org/abs/1106.5730">arXiv:1106.5730</a>;
 * Zheng et al., "Asynchronous SGD with Delay Compensation", <a href="https://arxiv.org/abs/1609.08326">arXiv:1609.08326</a>;
 * Zhang et al., "Staleness-aware Async-SGD", <a href="https://arxiv.org/abs/1511.05950">arXiv:1511.05950</a>.
 */
public final class AsyncSgd {

    /** A sparse gradient: coordinates and values, reused between calls. */
    public static final class Sparse {

        private int[] index;
        private float[] value;
        private int size;

        /**
         * Creates an empty gradient.
         *
         * @param capacity initial capacity
         */
        public Sparse(int capacity) {
            this.index = new int[Math.max(1, capacity)];
            this.value = new float[Math.max(1, capacity)];
        }

        /**
         * Adds a contribution.
         *
         * @param i the coordinate
         * @param v the value
         */
        public void add(int i, float v) {
            if (size == index.length) {
                index = Arrays.copyOf(index, size * 2);
                value = Arrays.copyOf(value, size * 2);
            }
            index[size] = i;
            value[size++] = v;
        }

        /** Empties the gradient. */
        public void clear() {
            size = 0;
        }

        /** @return the number of entries */
        public int size() {
            return size;
        }

        /**
         * @param k an entry
         * @return its coordinate
         */
        public int index(int k) {
            return index[k];
        }

        /**
         * @param k an entry
         * @return its value
         */
        public float value(int k) {
            return value[k];
        }
    }

    /** A learning problem over a flat parameter array. */
    public interface Problem {

        /** @return the number of parameters */
        int dimension();

        /** @return the number of training samples */
        int samples();

        /**
         * Computes a mini-batch's mean loss and gradient at {@code w}.
         *
         * @param w the parameters (may change concurrently under Hogwild)
         * @param batch the sample indices
         * @param out receives the gradient entries (a coordinate may repeat)
         * @return the loss
         */
        double gradient(float[] w, int[] batch, Sparse out);

        /**
         * The loss on held-out data.
         *
         * @param w the parameters
         * @return the loss
         */
        double evaluate(float[] w);
    }

    /**
     * A training curve.
     *
     * @param updates update counts at which the loss was measured
     * @param seconds wall time at those points
     * @param loss held-out loss at those points
     * @param finalLoss the loss at the end
     * @param wallSeconds total wall time
     */
    public record Curve(int[] updates, double[] seconds, double[] loss, double finalLoss, double wallSeconds) {
    }

    private AsyncSgd() {
    }

    /**
     * Hogwild!: {@code threads} workers each run {@code updatesPerThread} mini-batch updates on
     * a shared array with no synchronisation (or with a global lock).
     *
     * @param problem the problem
     * @param init the initial parameters (not modified)
     * @param threads workers
     * @param updatesPerThread updates per worker
     * @param batch mini-batch size
     * @param learningRate the step size
     * @param locked whether updates hold a global lock (the serialised baseline)
     * @param seed the seed
     * @return the final loss and wall time
     */
    public static Curve hogwild(Problem problem, float[] init, int threads, int updatesPerThread, int batch,
            float learningRate, boolean locked, long seed) {
        float[] w = init.clone();
        Object lock = new Object();
        long start = System.nanoTime();
        Collectives.run(threads, r -> {
            SplittableRandom random = new SplittableRandom(seed * 1_000_003L + r);
            Sparse g = new Sparse(64);
            int[] b = new int[batch];
            for (int t = 0; t < updatesPerThread; t++) {
                for (int i = 0; i < batch; i++) {
                    b[i] = random.nextInt(problem.samples());
                }
                g.clear();
                if (locked) {
                    synchronized (lock) {
                        problem.gradient(w, b, g);
                        apply(w, g, learningRate);
                    }
                } else {
                    problem.gradient(w, b, g);
                    apply(w, g, learningRate);
                }
            }
            return null;
        });
        double wall = (System.nanoTime() - start) / 1e9;
        double loss = problem.evaluate(w);
        return new Curve(new int[] {threads * updatesPerThread}, new double[] {wall}, new double[] {loss}, loss, wall);
    }

    private static void apply(float[] w, Sparse g, float lr) {
        for (int k = 0; k < g.size(); k++) {
            w[g.index(k)] -= lr * g.value(k);
        }
    }

    /**
     * Simulated asynchronous SGD with injected staleness.
     *
     * @param problem the problem
     * @param init the initial parameters (not modified)
     * @param updates the number of updates
     * @param batch mini-batch size
     * @param learningRate the step size
     * @param delay the staleness of update {@code t}, given {@code t}; capped at {@code t}
     * @param compensation DC-ASGD's {@code lambda}; 0 for plain ASGD; the adaptive variant
     *     (DC-ASGD-a) divides it by the root of a running mean of squared gradients per coordinate
     * @param stalenessAware whether the step is divided by {@code max(1, tau)}
     * @param evalEvery measure the held-out loss every this many updates
     * @param seed the seed
     * @return the curve
     */
    public static Curve simulate(Problem problem, float[] init, int updates, int batch, float learningRate,
            IntUnaryOperator delay, float compensation, boolean stalenessAware, int evalEvery, long seed) {
        float[] meanSquare = new float[problem.dimension()];
        int maxDelay = 0;
        for (int t = 0; t < Math.min(updates, 4096); t++) {
            maxDelay = Math.max(maxDelay, delay.applyAsInt(t));
        }
        int dim = problem.dimension();
        float[][] history = new float[maxDelay + 1][];
        float[] w = init.clone();
        history[0] = w.clone();
        SplittableRandom random = new SplittableRandom(seed);
        Sparse g = new Sparse(64);
        int[] b = new int[batch];
        List<Integer> at = new ArrayList<>();
        List<Double> secs = new ArrayList<>();
        List<Double> losses = new ArrayList<>();
        long start = System.nanoTime();
        float[] dense = new float[dim];
        boolean[] touched = new boolean[dim];
        int[] list = new int[dim];
        for (int t = 0; t < updates; t++) {
            int tau = Math.min(Math.min(delay.applyAsInt(t), t), maxDelay);
            float[] old = tau == 0 ? w : history[(t - tau) % history.length];
            for (int i = 0; i < batch; i++) {
                b[i] = random.nextInt(problem.samples());
            }
            g.clear();
            problem.gradient(old, b, g);
            // sum repeated coordinates so the compensation sees the full gradient
            int n = 0;
            for (int k = 0; k < g.size(); k++) {
                int i = g.index(k);
                if (!touched[i]) {
                    touched[i] = true;
                    list[n++] = i;
                }
                dense[i] += g.value(k);
            }
            float lr = stalenessAware ? learningRate / Math.max(1, tau) : learningRate;
            for (int k = 0; k < n; k++) {
                int i = list[k];
                float gi = dense[i];
                meanSquare[i] = 0.95f * meanSquare[i] + 0.05f * gi * gi;
                if (compensation != 0 && tau > 0) {
                    float lambda = compensation / (float) Math.sqrt(meanSquare[i] + 1e-7f);
                    gi += lambda * gi * gi * (w[i] - old[i]);
                }
                w[i] -= lr * gi;
                dense[i] = 0;
                touched[i] = false;
            }
            history[(t + 1) % history.length] = w.clone();
            if (evalEvery > 0 && ((t + 1) % evalEvery == 0 || t == updates - 1)) {
                at.add(t + 1);
                secs.add((System.nanoTime() - start) / 1e9);
                losses.add(problem.evaluate(w));
            }
        }
        double wall = (System.nanoTime() - start) / 1e9;
        double fin = problem.evaluate(w);
        return new Curve(at.stream().mapToInt(Integer::intValue).toArray(),
                secs.stream().mapToDouble(Double::doubleValue).toArray(),
                losses.stream().mapToDouble(Double::doubleValue).toArray(), fin, wall);
    }
}
