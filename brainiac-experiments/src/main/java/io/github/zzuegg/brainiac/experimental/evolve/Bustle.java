package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * BUSTLE: bottom-up synthesis in which a learned model reweights intermediate values. Each
 * value the search makes is described by a property signature, boolean properties of the
 * value against the target outputs (same length, contains the output's elements, a
 * constant offset, reversed, sorted alike and so on) aggregated over the examples as
 * all/some/none; an MLP maps the signature to the probability that the value lies on a path
 * to the solution, and {@link BottomUpSynthesis} adds {@code penalty * (1 - p)} to the
 * value's weight, so promising values are extended first.
 *
 * <p>The model is trained, as in the paper, on random programs: the intermediate values of
 * the generating program are positives and values one primitive off that path are
 * negatives.
 *
 * <p>Paper: Odena, Shi, Bieber, Singh, Sutton and Dai, "BUSTLE: Bottom-Up Program Synthesis
 * Through Learning-Guided Exploration" (ICLR 2021),
 * <a href="https://arxiv.org/abs/2007.14381">arXiv:2007.14381</a>.
 */
public final class Bustle {

    /** Properties per example, before the all/some/none aggregation. */
    public static final int PROPERTIES = 20;
    /** Length of a property signature. */
    public static final int SIGNATURE = 3 * PROPERTIES;

    private final BottomUpSynthesis search;
    private final Model model;
    private final double penalty;

    /**
     * Creates an untrained BUSTLE.
     *
     * @param search the bottom-up search
     * @param hidden the MLP's hidden width
     * @param penalty the weight added to a value the model calls useless
     * @param seed the seed
     */
    public Bustle(BottomUpSynthesis search, int hidden, double penalty, long seed) {
        this.search = search;
        this.penalty = penalty;
        this.model = Model.sequential().input(SIGNATURE).dense(hidden, Activation.RELU).dense(1).seed(seed).build();
    }

    /**
     * The property signature of a value against the target outputs.
     *
     * @param value one list per example
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @return {@link #SIGNATURE} values: per property, whether it holds on all, some or none
     *     of the examples (one-hot)
     */
    public static float[] signature(int[][] value, int[][] inputs, int[][] outputs) {
        int[] count = new int[PROPERTIES];
        for (int e = 0; e < value.length; e++) {
            boolean[] p = properties(value[e], inputs[e], outputs[e]);
            for (int k = 0; k < PROPERTIES; k++) {
                count[k] += p[k] ? 1 : 0;
            }
        }
        float[] s = new float[SIGNATURE];
        for (int k = 0; k < PROPERTIES; k++) {
            int slot = count[k] == value.length ? 0 : count[k] > 0 ? 1 : 2;
            s[3 * k + slot] = 1;
        }
        return s;
    }

    private static boolean[] properties(int[] v, int[] in, int[] o) {
        boolean[] p = new boolean[PROPERTIES];
        boolean sameLen = v.length == o.length;
        p[0] = Arrays.equals(v, o);
        p[1] = sameLen;
        p[2] = v.length > o.length;
        p[3] = v.length < o.length;
        p[4] = subMultiset(o, v);
        p[5] = subSet(o, v);
        int[] sv = v.clone();
        int[] so = o.clone();
        Arrays.sort(sv);
        Arrays.sort(so);
        p[6] = Arrays.equals(sv, so);
        p[7] = contiguous(o, v);
        p[8] = contiguous(v, o);
        boolean offset = sameLen && v.length > 0;
        int factor = 0;
        for (int i = 0; i < v.length && sameLen && factor == 0; i++) {
            factor = v[i] != 0 && o[i] % v[i] == 0 ? o[i] / v[i] : 0;
        }
        boolean ratio = sameLen && factor > 1;
        boolean parity = sameLen;
        boolean reversed = sameLen;
        for (int i = 0; i < v.length && sameLen; i++) {
            offset &= o[i] - v[i] == o[0] - v[0];
            ratio &= o[i] == factor * v[i];
            parity &= (o[i] - v[i]) % 2 == 0;
            reversed &= v[i] == o[o.length - 1 - i];
        }
        p[9] = offset;
        p[10] = ratio;
        p[11] = v.length > 0 && o.length > 0 && v[0] == o[0];
        p[12] = v.length > 0 && o.length > 0 && v[v.length - 1] == o[o.length - 1];
        p[13] = reversed;
        p[14] = sorted(v) == sorted(o);
        p[15] = Arrays.equals(v, in);
        p[16] = parity;
        p[17] = v.length > 0 && o.length > 0 && Arrays.stream(v).max().getAsInt() == Arrays.stream(o).max().getAsInt();
        p[18] = subsequence(v, o);
        p[19] = subsequence(o, v);
        return p;
    }

    private static boolean sorted(int[] xs) {
        for (int i = 1; i < xs.length; i++) {
            if (xs[i] < xs[i - 1]) {
                return false;
            }
        }
        return true;
    }

    private static boolean subMultiset(int[] small, int[] big) {
        int[] a = small.clone();
        int[] b = big.clone();
        Arrays.sort(a);
        Arrays.sort(b);
        int j = 0;
        for (int x : a) {
            while (j < b.length && b[j] < x) {
                j++;
            }
            if (j == b.length || b[j] != x) {
                return false;
            }
            j++;
        }
        return true;
    }

    private static boolean subSet(int[] small, int[] big) {
        Set<Integer> s = new HashSet<>();
        for (int x : big) {
            s.add(x);
        }
        for (int x : small) {
            if (!s.contains(x)) {
                return false;
            }
        }
        return true;
    }

    private static boolean contiguous(int[] small, int[] big) {
        outer:
        for (int i = 0; i + small.length <= big.length; i++) {
            for (int k = 0; k < small.length; k++) {
                if (big[i + k] != small[k]) {
                    continue outer;
                }
            }
            return true;
        }
        return false;
    }

    private static boolean subsequence(int[] small, int[] big) {
        int j = 0;
        for (int x : big) {
            if (j < small.length && small[j] == x) {
                j++;
            }
        }
        return j == small.length;
    }

    /**
     * Trains the usefulness model on random programs (binary cross-entropy).
     *
     * @param source generates the random programs and examples
     * @param steps training steps
     * @param batch programs per step
     * @param maxLength the longest training program
     * @param examples examples per program
     * @param learningRate the Adam rate
     * @param random the source
     * @return the last step's accuracy at threshold 0.5
     */
    public double train(GuidedEnumerator source, int steps, int batch, int maxLength, int examples, float learningRate,
            SplittableRandom random) {
        Optimizer opt = Optimizer.adam(learningRate);
        List<String> prims = search.primitives();
        double acc = 0;
        for (int s = 0; s < steps; s++) {
            List<float[]> xs = new ArrayList<>();
            List<Float> ys = new ArrayList<>();
            int made = 0;
            while (made < batch) {
                Object[] t = source.randomTask(1 + random.nextInt(maxLength), examples, random);
                if (t == null) {
                    continue;
                }
                made++;
                int[] idx = (int[]) t[0];
                int[][] in = (int[][]) t[1];
                int[][] out = (int[][]) t[2];
                Set<String> path = new HashSet<>();
                int[][] v = in;
                List<int[][]> trace = new ArrayList<>();
                trace.add(in);
                for (int i : idx) {
                    v = search.apply(prims.get(i), v);
                    trace.add(v);
                    path.add(Arrays.deepToString(v));
                }
                for (int k = 1; k < trace.size() - 1; k++) {
                    xs.add(signature(trace.get(k), in, out));
                    ys.add(1f);
                }
                for (int k = 0; k < trace.size() - 1; k++) {
                    for (int tries = 0; tries < 2; tries++) {
                        int[][] off = search.apply(prims.get(random.nextInt(prims.size())), trace.get(k));
                        if (off != null && !path.contains(Arrays.deepToString(off))) {
                            xs.add(signature(off, in, out));
                            ys.add(0f);
                        }
                    }
                }
            }
            int b = xs.size();
            float[] x = new float[b * SIGNATURE];
            float[] y = new float[b];
            for (int i = 0; i < b; i++) {
                System.arraycopy(xs.get(i), 0, x, i * SIGNATURE, SIGNATURE);
                y[i] = ys.get(i);
            }
            Var z = model.forward(Var.of(Tensor.of(Shape.of(b, SIGNATURE), x)), true).reshape(b);
            float[] zv = z.value().toArray();
            int right = 0;
            for (int i = 0; i < b; i++) {
                right += (zv[i] > 0) == (y[i] > 0.5f) ? 1 : 0;
            }
            acc = right / (double) b;
            Var loss = z.softplus().sub(z.mul(Tensor.of(Shape.of(b), y))).mean();
            loss.backward();
            opt.step(model.parameters());
        }
        return acc;
    }

    /**
     * The model's usefulness probability of a value.
     *
     * @param value one list per example
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @return the probability that the value is on a path to the solution
     */
    public double usefulness(int[][] value, int[][] inputs, int[][] outputs) {
        float[] s = signature(value, inputs, outputs);
        return Var.noGrad(() -> model.forward(Var.of(Tensor.of(Shape.of(1, SIGNATURE), s)), false).sigmoid())
                .value().toArray()[0];
    }

    /**
     * The reweighted search.
     *
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @param maxLength the longest program
     * @param budget the most primitive applications
     * @return the outcome
     */
    public BottomUpSynthesis.Result search(int[][] inputs, int[][] outputs, int maxLength, long budget) {
        return search.search(inputs, outputs, maxLength, null,
                (v, in, out) -> penalty * (1 - usefulness(v, in, out)), null, budget);
    }
}
