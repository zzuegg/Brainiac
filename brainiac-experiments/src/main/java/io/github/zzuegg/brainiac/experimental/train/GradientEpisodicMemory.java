package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.BiFunction;

/**
 * Gradient episodic memory: keep a few examples of every finished task and never take a
 * step that would increase their loss. Before each optimizer step the new-task gradient
 * {@code g} is compared with the gradients {@code g_k} of the memories:
 *
 * <ul>
 *   <li><b>GEM</b> (Lopez-Paz and Ranzato 2017, <a href="https://arxiv.org/abs/1706.08840">arXiv:1706.08840</a>):
 *       one constraint per past task, {@code g~ . g_k >= 0}. The closest such gradient,
 *       {@code argmin ||g~ - g||^2}, comes from the small dual quadratic program
 *       {@code min_v 1/2 v^T G G^T v + g^T G^T v, v >= 0} (one variable per task, solved
 *       here by projected coordinate descent), then {@code g~ = g + G^T v}.</li>
 *   <li><b>A-GEM</b> (Chaudhry et al. 2019, <a href="https://arxiv.org/abs/1812.00420">arXiv:1812.00420</a>):
 *       one averaged constraint on a random batch from all memories; if {@code g . g_ref < 0}
 *       the conflicting component is projected out,
 *       {@code g~ = g - (g . g_ref / g_ref . g_ref) g_ref}. One extra backward pass per step
 *       however many tasks there are.</li>
 * </ul>
 *
 * <p>Use it as a drop-in {@link Optimizer} decorator ({@link #wrap}): the wrapped optimizer
 * sees the projected gradients. Call {@link #remember} after each task.
 */
public final class GradientEpisodicMemory {

    private final List<Var> parameters;
    private final BiFunction<Tensor, Tensor, Var> loss;
    private final int perTask;
    private final boolean averaged;
    private final int referenceBatch;
    private final SplittableRandom random;
    private final List<Tensor[]> memories = new ArrayList<>();
    private long projections;
    private long steps;

    /**
     * Creates the memory.
     *
     * @param parameters the parameters being trained
     * @param loss the training loss of inputs and class ids, with the model's current weights
     * @param perTask examples kept per finished task
     * @param averaged A-GEM's single averaged constraint instead of GEM's per-task ones
     * @param referenceBatch A-GEM's memory batch size
     * @param seed seeds the memory selection
     */
    public GradientEpisodicMemory(List<Var> parameters, BiFunction<Tensor, Tensor, Var> loss, int perTask,
            boolean averaged, int referenceBatch, long seed) {
        this.parameters = List.copyOf(parameters);
        this.loss = loss;
        this.perTask = perTask;
        this.averaged = averaged;
        this.referenceBatch = referenceBatch;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Stores random examples of a finished task.
     *
     * @param x the task's inputs {@code [n, features]}
     * @param ids the task's class ids {@code [n]}
     */
    public void remember(Tensor x, Tensor ids) {
        int n = x.shape().dim(0);
        int[] rows = random.ints(0, n).distinct().limit(Math.min(perTask, n)).toArray();
        memories.add(new Tensor[] {take(x, rows), take(ids, rows)});
    }

    private static Tensor take(Tensor t, int[] rows) {
        Tensor[] parts = new Tensor[rows.length];
        for (int i = 0; i < rows.length; i++) {
            parts[i] = t.slice(0, rows[i], rows[i] + 1);
        }
        return Tensor.concat(0, parts);
    }

    /** @return the share of steps whose gradient had to be projected */
    public double projectionRate() {
        return steps == 0 ? 0 : projections / (double) steps;
    }

    /**
     * Wraps an optimizer so that every step first projects the gradients.
     *
     * @param inner the optimizer that applies the projected gradients
     * @return the decorated optimizer
     */
    public Optimizer wrap(Optimizer inner) {
        return new Optimizer() {
            @Override
            public void step(List<Var> ps) {
                project();
                inner.step(ps);
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
            public void forget(Var parameter) {
                inner.forget(parameter);
            }

            @Override
            public String toString() {
                return (averaged ? "A-GEM(" : "GEM(") + inner + ")";
            }
        };
    }

    private void project() {
        steps++;
        if (memories.isEmpty()) {
            return;
        }
        Tensor[] g = new Tensor[parameters.size()];
        for (int i = 0; i < g.length; i++) {
            Var p = parameters.get(i);
            g[i] = p.grad() == null ? Tensor.zeros(p.shape()) : p.grad();
        }
        List<Tensor[]> refs = new ArrayList<>();
        if (averaged) {
            int[] counts = new int[memories.size()];
            for (int i = 0; i < referenceBatch; i++) {
                counts[random.nextInt(memories.size())]++;
            }
            List<Tensor> xs = new ArrayList<>();
            List<Tensor> ys = new ArrayList<>();
            for (int m = 0; m < counts.length; m++) {
                Tensor[] mem = memories.get(m);
                int n = mem[0].shape().dim(0);
                int[] rows = random.ints(counts[m], 0, n).toArray();
                if (rows.length > 0) {
                    xs.add(take(mem[0], rows));
                    ys.add(take(mem[1], rows));
                }
            }
            refs.add(gradient(Tensor.concat(0, xs.toArray(new Tensor[0])), Tensor.concat(0, ys.toArray(new Tensor[0]))));
        } else {
            for (Tensor[] mem : memories) {
                refs.add(gradient(mem[0], mem[1]));
            }
        }
        int k = refs.size();
        double[] gg = new double[k];
        double[][] gram = new double[k][k];
        for (int a = 0; a < k; a++) {
            gg[a] = dot(g, refs.get(a));
            for (int b = 0; b <= a; b++) {
                gram[a][b] = dot(refs.get(a), refs.get(b));
                gram[b][a] = gram[a][b];
            }
        }
        boolean violated = false;
        for (double v : gg) {
            violated |= v < 0;
        }
        Tensor[] out = g;
        if (violated) {
            projections++;
            double[] v = new double[k];
            if (averaged) {
                v[0] = -gg[0] / Math.max(gram[0][0], 1e-12);
            } else {
                // dual QP by projected coordinate descent: min 1/2 v'Mv + gg'v, v >= 0
                for (int sweep = 0; sweep < 100; sweep++) {
                    for (int a = 0; a < k; a++) {
                        double r = gg[a];
                        for (int b = 0; b < k; b++) {
                            r += b == a ? 0 : gram[a][b] * v[b];
                        }
                        v[a] = Math.max(0, -r / Math.max(gram[a][a], 1e-12));
                    }
                }
            }
            out = new Tensor[g.length];
            for (int i = 0; i < g.length; i++) {
                Tensor t = g[i];
                for (int a = 0; a < k; a++) {
                    if (v[a] != 0) {
                        t = t.add(refs.get(a)[i].mul((float) v[a]));
                    }
                }
                out[i] = t;
            }
        }
        for (int i = 0; i < g.length; i++) {
            parameters.get(i).setGrad(out[i]);
        }
    }

    private Tensor[] gradient(Tensor x, Tensor ids) {
        List<Var> grads = Var.gradients(loss.apply(x, ids), parameters, false);
        Tensor[] out = new Tensor[grads.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = grads.get(i).value();
        }
        return out;
    }

    private static double dot(Tensor[] a, Tensor[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i].mul(b[i]).sum().item();
        }
        return s;
    }

    /**
     * The memory sizes, for logging.
     *
     * @return tasks remembered and examples per task
     */
    public Map<String, Integer> sizes() {
        return Map.of("tasks", memories.size(), "perTask", perTask);
    }
}
