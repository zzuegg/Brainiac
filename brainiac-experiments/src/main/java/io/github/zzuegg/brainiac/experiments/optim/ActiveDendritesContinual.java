package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ActiveDendriteLayer;
import io.github.zzuegg.brainiac.experimental.train.ForgettingTracker;
import io.github.zzuegg.brainiac.experimental.train.SynapticIntelligence;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Active dendrites against catastrophic forgetting on the permuted-digits sequence of
 * {@link ContinualLearningExperiment}, in the paper's setting: one shared ten-way output head
 * for all tasks and a context vector per task (Iyer et al.
 * <a href="https://arxiv.org/abs/2201.00042">arXiv:2201.00042</a>), here the one-hot task
 * identity. (The paper's alternative, each task's mean input as context, works poorly on these
 * digits: permuted prototypes of non-negative pixels are too similar to open different
 * subnetworks.) Each method trains on the tasks in order and is scored with a
 * {@link ForgettingTracker}.
 *
 * <ul>
 *   <li>{@link Method#DENDRITES}: two {@link ActiveDendriteLayer}s of 256 units (k-WTA at 5% density)
 *       and a linear head;</li>
 *   <li>{@link Method#MLP}: a plain ReLU MLP of the same width, fine-tuned (the forgetting baseline);</li>
 *   <li>{@link Method#MLP_CONTEXT}: the MLP with the same context concatenated to its input, to
 *       separate the effect of the dendritic gating from that of the context itself;</li>
 *   <li>{@link Method#MLP_SI}: the MLP protected by {@link SynapticIntelligence}.</li>
 * </ul>
 * EWC and frozen growing regions run through {@link ContinualLearningExperiment#run}, whose
 * {@code ContinualBrain} uses one head per task (an easier, task-incremental protocol).
 */
public final class ActiveDendritesContinual {

    /** The compared methods. */
    public enum Method {
        /** active dendrites with k-WTA */
        DENDRITES,
        /** a plain MLP */
        MLP,
        /** an MLP that also sees the context */
        MLP_CONTEXT,
        /** an MLP with synaptic intelligence */
        MLP_SI
    }

    private static final int HIDDEN = 256;
    private static final float DENSITY = 0.05f;

    private ActiveDendritesContinual() {
    }

    private static Tensor withContext(ToyTasks.Data d, float[] context) {
        int f = d.features();
        float[] v = new float[d.size() * (f + context.length)];
        for (int i = 0; i < d.size(); i++) {
            System.arraycopy(d.x()[i], 0, v, i * (f + context.length), f);
            System.arraycopy(context, 0, v, i * (f + context.length) + f, context.length);
        }
        return Tensor.of(Shape.of(d.size(), f + context.length), v);
    }

    /**
     * Trains one method on the tasks in order.
     *
     * @param tasks the tasks
     * @param method the method
     * @param epochs epochs per task
     * @param seed the seed
     * @return the accuracy matrix
     */
    public static ForgettingTracker run(List<ContinualLearningExperiment.Task> tasks, Method method, int epochs,
            long seed) {
        int f = tasks.get(0).train().features();
        boolean context = method == Method.DENDRITES || method == Method.MLP_CONTEXT;
        Model model = switch (method) {
            case DENDRITES -> Model.sequential().input(f + tasks.size())
                    .add(ActiveDendriteLayer.spec(tasks.size(), HIDDEN, tasks.size(), DENSITY, false))
                    .add(ActiveDendriteLayer.spec(tasks.size(), HIDDEN, tasks.size(), DENSITY, true))
                    .dense(10).seed(seed).build();
            case MLP_CONTEXT -> Model.sequential().input(f + tasks.size()).dense(HIDDEN, Activation.RELU)
                    .dense(HIDDEN, Activation.RELU).dense(10).seed(seed).build();
            default -> Model.sequential().input(f).dense(HIDDEN, Activation.RELU).dense(HIDDEN, Activation.RELU)
                    .dense(10).seed(seed).build();
        };
        SynapticIntelligence si = method == Method.MLP_SI ? new SynapticIntelligence(model.parameters(), 1f, 0.1f)
                : null;
        float lr = 0.01f;
        Optimizer opt = si == null ? Optimizer.adam(lr) : si.track(Optimizer.adam(lr));
        float[][] contexts = new float[tasks.size()][];
        for (int t = 0; t < tasks.size(); t++) {
            contexts[t] = new float[context ? tasks.size() : 0];
            if (context) {
                contexts[t][t] = 1;
            }
        }
        ForgettingTracker tracker = new ForgettingTracker();
        SplittableRandom r = new SplittableRandom(seed);
        for (int t = 0; t < tasks.size(); t++) {
            ToyTasks.Data train = tasks.get(t).train();
            Tensor x = context ? withContext(train, contexts[t]) : train.inputs();
            Tensor y = train.ids();
            int n = train.size();
            for (int e = 0; e < epochs; e++) {
                for (int from = 0; from + 32 <= n; from += 32) {
                    int start = Math.min(n - 32, from + r.nextInt(8));
                    Var loss = Functional.crossEntropy(model.forward(Var.of(x.slice(0, start, start + 32)), true),
                            y.slice(0, start, start + 32));
                    if (si != null) {
                        loss = loss.add(si.penalty());
                    }
                    loss.backward();
                    opt.step(model.parameters());
                }
            }
            if (si != null) {
                si.consolidate();
            }
            double[] acc = new double[t + 1];
            for (int j = 0; j <= t; j++) {
                ToyTasks.Data test = tasks.get(j).test();
                Tensor tx = context ? withContext(test, contexts[j]) : test.inputs();
                acc[j] = ToyTasks.accuracy(model.forward(Var.of(tx), false).value(), test.y());
            }
            tracker.record(acc);
        }
        return tracker;
    }
}
