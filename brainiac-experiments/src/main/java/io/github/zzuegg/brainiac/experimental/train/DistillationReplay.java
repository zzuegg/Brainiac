package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.TreeSet;

/**
 * Continual learning by distillation and replay, for class-incremental learning with one
 * shared classifier head. Three classic methods share this class:
 *
 * <ul>
 *   <li><b>LwF</b> (Learning without Forgetting, Li and Hoiem 2016,
 *       <a href="https://arxiv.org/abs/1606.09282">arXiv:1606.09282</a>): no memory at all.
 *       Before a new task, the current model is frozen as a teacher; while learning the new
 *       task, the student's outputs for the old classes on the <em>new</em> data are
 *       distilled toward the teacher's (temperature-softened KL).</li>
 *   <li><b>iCaRL</b> (Rebuffi et al. 2017, <a href="https://arxiv.org/abs/1611.07725">arXiv:1611.07725</a>):
 *       a fixed budget of exemplars split evenly over the classes seen, chosen by
 *       <em>herding</em> (greedily matching the class's mean feature), replayed with the new
 *       data plus LwF distillation; prediction is the <b>nearest mean of exemplars</b> in
 *       normalized feature space instead of the biased linear head.</li>
 *   <li><b>DER++</b> (Dark Experience Replay, Buzzega et al. 2020,
 *       <a href="https://arxiv.org/abs/2004.07211">arXiv:2004.07211</a>): a reservoir buffer
 *       that stores each sample's <em>logits</em> at the time it was seen; replay matches
 *       those logits (MSE, weight {@code alpha}) and the labels (cross-entropy, weight
 *       {@code beta}).</li>
 * </ul>
 *
 * <p>Works with any sequential {@link Model} whose last layer is the classifier head; iCaRL
 * uses the output of the layers before it as features. Call {@link #loss} for each training
 * batch and {@link #finishTask} after each task.
 */
public final class DistillationReplay {

    /** The method. */
    public enum Kind {
        /** Learning without Forgetting. */
        LWF,
        /** Incremental classifier and representation learning. */
        ICARL,
        /** Dark experience replay with labels. */
        DER_PLUS_PLUS
    }

    private final Kind kind;
    private final Model model;
    private final int memory;
    private final float alpha;
    private final float beta;
    private final float temperature;
    private final SplittableRandom random;
    private final TreeSet<Integer> oldClasses = new TreeSet<>();
    private final List<float[]> bufferX = new ArrayList<>();
    private final List<Integer> bufferY = new ArrayList<>();
    private final List<float[]> bufferLogits = new ArrayList<>();
    private Model teacher;
    private long seen;
    private Tensor classMeans;
    private int[] meanClasses;

    /**
     * Creates the learner.
     *
     * @param kind the method
     * @param model the model being trained; its last layer is the head
     * @param memory the exemplar or buffer budget in samples (ignored by LwF)
     * @param alpha the distillation (LwF, iCaRL) or logit-matching (DER++) weight
     * @param beta the replayed-label weight of DER++
     * @param temperature the distillation temperature
     * @param seed seeds the buffer sampling
     */
    public DistillationReplay(Kind kind, Model model, int memory, float alpha, float beta, float temperature,
            long seed) {
        this.kind = kind;
        this.model = model;
        this.memory = memory;
        this.alpha = alpha;
        this.beta = beta;
        this.temperature = temperature;
        this.random = new SplittableRandom(seed);
    }

    /** @return the samples held in memory */
    public int stored() {
        return bufferY.size();
    }

    /**
     * The training loss of one batch of the current task, including distillation and replay.
     *
     * @param x inputs {@code [batch, features]}
     * @param ids class ids {@code [batch]}
     * @return the loss to back-propagate
     */
    public Var loss(Tensor x, Tensor ids) {
        Var logits = model.forward(Var.of(x), true);
        Var loss = Functional.crossEntropy(logits, ids);
        if (kind == Kind.DER_PLUS_PLUS) {
            loss = loss.add(darkReplay());
            remember(x, ids, logits.value());
        } else {
            if (kind == Kind.ICARL && stored() > 0) {
                Tensor[] ex = sample(Math.min(stored(), x.shape().dim(0)), false);
                loss = loss.add(Functional.crossEntropy(model.forward(Var.of(ex[0]), true), ex[1]));
                loss = loss.add(distill(ex[0]));
            }
            loss = loss.add(distill(x));
        }
        return loss;
    }

    private Var distill(Tensor x) {
        if (teacher == null || oldClasses.isEmpty()) {
            return Var.of(Tensor.scalar(0));
        }
        int[] old = oldClasses.stream().mapToInt(Integer::intValue).toArray();
        Tensor pick = selector(model.outputSize(), old);
        Var student = model.forward(Var.of(x), true).matmul(Var.of(pick)).div(temperature);
        Tensor t = teacher.forward(Var.of(x)).value().matmul(pick).div(temperature);
        Var target = Activation.SOFTMAX.forward(Var.of(t));
        Var logp = Activation.LOG_SOFTMAX.forward(student);
        // KL(teacher || student) up to the teacher's constant entropy, scaled by T^2
        return logp.mul(target).sum(-1, false).mean().mul(-alpha * temperature * temperature);
    }

    private static Tensor selector(int classes, int[] keep) {
        float[] s = new float[classes * keep.length];
        for (int j = 0; j < keep.length; j++) {
            s[keep[j] * keep.length + j] = 1;
        }
        return Tensor.of(Shape.of(classes, keep.length), s);
    }

    private Var darkReplay() {
        if (stored() == 0) {
            return Var.of(Tensor.scalar(0));
        }
        int n = Math.min(stored(), 32);
        Tensor[] a = sample(n, true);
        Var logitLoss = model.forward(Var.of(a[0]), true).sub(a[2]).square().mean().mul(alpha);
        Tensor[] b = sample(n, false);
        return logitLoss.add(Functional.crossEntropy(model.forward(Var.of(b[0]), true), b[1]).mul(beta));
    }

    /** {@code {x, ids, logits?}} of random buffer rows. */
    private Tensor[] sample(int n, boolean logits) {
        int f = bufferX.get(0).length;
        float[] x = new float[n * f];
        float[] y = new float[n];
        int k = logits ? bufferLogits.get(0).length : 0;
        float[] l = new float[n * k];
        for (int i = 0; i < n; i++) {
            int j = random.nextInt(stored());
            System.arraycopy(bufferX.get(j), 0, x, i * f, f);
            y[i] = bufferY.get(j);
            if (logits) {
                System.arraycopy(bufferLogits.get(j), 0, l, i * k, k);
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, f), x), Tensor.of(Shape.of(n), y),
            logits ? Tensor.of(Shape.of(n, k), l) : null};
    }

    /** Reservoir sampling of DER++'s buffer. */
    private void remember(Tensor x, Tensor ids, Tensor logits) {
        int n = x.shape().dim(0);
        int f = x.shape().dim(1);
        int k = logits.shape().dim(1);
        float[] xs = x.toArray();
        float[] ys = ids.toArray();
        float[] ls = logits.toArray();
        for (int i = 0; i < n; i++) {
            seen++;
            int slot = stored() < memory ? stored() : (int) random.nextLong(seen);
            if (slot >= memory) {
                continue;
            }
            float[] row = new float[f];
            System.arraycopy(xs, i * f, row, 0, f);
            float[] lr = new float[k];
            System.arraycopy(ls, i * k, lr, 0, k);
            if (slot == stored()) {
                bufferX.add(row);
                bufferY.add((int) ys[i]);
                bufferLogits.add(lr);
            } else {
                bufferX.set(slot, row);
                bufferY.set(slot, (int) ys[i]);
                bufferLogits.set(slot, lr);
            }
        }
    }

    /**
     * Ends a task: freezes the teacher (LwF, iCaRL), and for iCaRL re-balances the exemplar
     * memory by herding and recomputes the class means.
     *
     * @param x the task's training inputs
     * @param ids the task's class ids
     */
    public void finishTask(Tensor x, Tensor ids) {
        float[] y = ids.toArray();
        TreeSet<Integer> classes = new TreeSet<>();
        for (float v : y) {
            classes.add((int) v);
        }
        if (kind == Kind.ICARL) {
            herd(x, y, classes);
        }
        oldClasses.addAll(classes);
        teacher = model.copy(model.device());
    }

    /** iCaRL's exemplar management: shrink old classes' sets, add herded sets for the new ones. */
    private void herd(Tensor x, float[] y, TreeSet<Integer> newClasses) {
        int total = oldClasses.size() + newClasses.size();
        int per = Math.max(1, memory / total);
        List<float[]> keepX = new ArrayList<>();
        List<Integer> keepY = new ArrayList<>();
        // exemplar sets are herding-ordered, so shrinking keeps the first ones
        int[] count = new int[model.outputSize()];
        for (int i = 0; i < stored(); i++) {
            int c = bufferY.get(i);
            if (count[c]++ < per) {
                keepX.add(bufferX.get(i));
                keepY.add(c);
            }
        }
        int f = x.shape().dim(1);
        float[] xs = x.toArray();
        for (int c : newClasses) {
            List<float[]> rows = new ArrayList<>();
            for (int i = 0; i < y.length; i++) {
                if ((int) y[i] == c) {
                    float[] r = new float[f];
                    System.arraycopy(xs, i * f, r, 0, f);
                    rows.add(r);
                }
            }
            double[][] feats = normalized(features(Tensor.of(rows.toArray(new float[0][]))));
            int d = feats[0].length;
            double[] mean = new double[d];
            for (double[] r : feats) {
                for (int j = 0; j < d; j++) {
                    mean[j] += r[j] / feats.length;
                }
            }
            double[] sum = new double[d];
            boolean[] used = new boolean[feats.length];
            for (int k = 1; k <= Math.min(per, feats.length); k++) {
                int best = -1;
                double bestDist = Double.POSITIVE_INFINITY;
                for (int i = 0; i < feats.length; i++) {
                    if (used[i]) {
                        continue;
                    }
                    double dist = 0;
                    for (int j = 0; j < d; j++) {
                        double v = mean[j] - (sum[j] + feats[i][j]) / k;
                        dist += v * v;
                    }
                    if (dist < bestDist) {
                        bestDist = dist;
                        best = i;
                    }
                }
                used[best] = true;
                for (int j = 0; j < d; j++) {
                    sum[j] += feats[best][j];
                }
                keepX.add(rows.get(best));
                keepY.add(c);
            }
        }
        bufferX.clear();
        bufferY.clear();
        bufferX.addAll(keepX);
        bufferY.addAll(keepY);
        updateMeans();
    }

    private void updateMeans() {
        TreeSet<Integer> classes = new TreeSet<>(bufferY);
        meanClasses = classes.stream().mapToInt(Integer::intValue).toArray();
        double[][] feats = normalized(features(Tensor.of(bufferX.toArray(new float[0][]))));
        int d = feats[0].length;
        float[] means = new float[meanClasses.length * d];
        for (int k = 0; k < meanClasses.length; k++) {
            int n = 0;
            double[] m = new double[d];
            for (int i = 0; i < feats.length; i++) {
                if (bufferY.get(i) == meanClasses[k]) {
                    n++;
                    for (int j = 0; j < d; j++) {
                        m[j] += feats[i][j];
                    }
                }
            }
            double norm = 0;
            for (double v : m) {
                norm += v * v;
            }
            norm = Math.sqrt(norm) + 1e-12;
            for (int j = 0; j < d; j++) {
                means[k * d + j] = (float) (m[j] / norm);
            }
        }
        classMeans = Tensor.of(Shape.of(meanClasses.length, d), means);
    }

    /** The output of every layer but the head. */
    private Tensor features(Tensor x) {
        List<Layer> layers = model.layers();
        Var h = Var.of(x);
        for (int i = 0; i < layers.size() - 1; i++) {
            h = layers.get(i).forward(h, false);
        }
        return h.value();
    }

    private static double[][] normalized(Tensor t) {
        int n = t.shape().dim(0);
        int d = t.shape().dim(1);
        float[] v = t.toArray();
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            double norm = 0;
            for (int j = 0; j < d; j++) {
                norm += (double) v[i * d + j] * v[i * d + j];
            }
            norm = Math.sqrt(norm) + 1e-12;
            for (int j = 0; j < d; j++) {
                out[i][j] = v[i * d + j] / norm;
            }
        }
        return out;
    }

    /**
     * Classifies inputs: nearest mean of exemplars for iCaRL (once it has exemplars), the
     * head's argmax otherwise.
     *
     * @param x inputs {@code [n, features]}
     * @return the class of each row
     */
    public int[] classify(Tensor x) {
        if (kind == Kind.ICARL && classMeans != null) {
            double[][] f = normalized(features(x));
            float[] m = classMeans.toArray();
            int d = f[0].length;
            int[] out = new int[f.length];
            for (int i = 0; i < f.length; i++) {
                double best = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < meanClasses.length; k++) {
                    double dot = 0;
                    for (int j = 0; j < d; j++) {
                        dot += f[i][j] * m[k * d + j];
                    }
                    if (dot > best) {
                        best = dot;
                        out[i] = meanClasses[k];
                    }
                }
            }
            return out;
        }
        Tensor logits = model.forward(Var.of(x)).value();
        int k = logits.shape().dim(1);
        float[] l = logits.toArray();
        int[] out = new int[x.shape().dim(0)];
        for (int i = 0; i < out.length; i++) {
            for (int c = 1; c < k; c++) {
                if (l[i * k + c] > l[i * k + out[i]]) {
                    out[i] = c;
                }
            }
        }
        return out;
    }
}
