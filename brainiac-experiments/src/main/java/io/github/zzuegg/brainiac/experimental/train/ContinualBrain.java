package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * A continual learner built on a growing {@link Brain} (experiment X21): every new task
 * wakes a <em>new region</em> that reads the input and, through sparse lateral
 * projections, the regions of all earlier tasks, and gets its own classifier head (as in
 * progressive networks, Rusu et al. 2016, <a href="https://arxiv.org/abs/1606.04671">arXiv:1606.04671</a>).
 * Earlier regions never read later ones, so protecting them protects old tasks:
 *
 * <ul>
 *   <li>{@link Protection#GROW_FROZEN}: old regions and heads are frozen, so old tasks
 *       cannot be forgotten at all, while new tasks reuse old features.</li>
 *   <li>{@link Protection#GROW_EWC}: old regions keep learning but are anchored by
 *       {@link ElasticWeightConsolidation}, trading a little forgetting for more
 *       transfer.</li>
 *   <li>{@link Protection#SHARED} and {@link Protection#SHARED_EWC}: baselines with one
 *       shared region for all tasks, plain fine-tuning and EWC.</li>
 * </ul>
 *
 * <p>All regions exist from the start, dormant; a task wakes its region with
 * {@link Brain#addNeurons}, so parameter identities never change and optimizers and EWC
 * anchors stay valid. Measure forgetting with a {@link ForgettingTracker}.
 */
public final class ContinualBrain {

    /** How old tasks are protected. */
    public enum Protection {
        /** one region for all tasks, plain fine-tuning (the forgetting baseline) */
        SHARED,
        /** one region for all tasks, protected by EWC */
        SHARED_EWC,
        /** a new region per task, old regions frozen */
        GROW_FROZEN,
        /** a new region per task, old regions protected by EWC */
        GROW_EWC
    }

    private final Protection protection;
    private final int regionSize;
    private final int maxTasks;
    private final Brain brain;
    private final List<Var[]> heads = new ArrayList<>();
    private final List<Integer> headInputs = new ArrayList<>();
    private final SplittableRandom random;
    private final ElasticWeightConsolidation ewc;

    /**
     * @param inputs input features
     * @param regionSize neurons per region
     * @param maxTasks the most tasks (regions) it can hold
     * @param lateralDensity the density of projections from old regions to new ones
     * @param protection how old tasks are protected
     * @param ewcLambda the EWC strength, for the EWC modes
     * @param seed the seed
     */
    public ContinualBrain(int inputs, int regionSize, int maxTasks, double lateralDensity, Protection protection,
            float ewcLambda, long seed) {
        this.protection = protection;
        this.regionSize = regionSize;
        this.maxTasks = maxTasks;
        this.random = new SplittableRandom(seed);
        boolean grow = grows();
        Brain.Builder b = Brain.builder();
        int regions = grow ? maxTasks : 1;
        for (int i = 0; i < regions; i++) {
            b.region("task" + i, regionSize).initialNeurons(grow ? 0 : regionSize).activation(Activation.TANH);
        }
        for (int i = 0; i < regions; i++) {
            b.connect(Brain.INPUT, "task" + i, 1);
            for (int j = 0; j < i; j++) {
                b.connect("task" + j, "task" + i, lateralDensity);
            }
        }
        String[] outs = new String[regions];
        for (int i = 0; i < regions; i++) {
            outs[i] = "task" + i;
        }
        brain = b.output(outs).build().create(Shape.of(inputs), new SplittableRandom(seed + 1));
        ewc = protection == Protection.SHARED_EWC || protection == Protection.GROW_EWC
                ? new ElasticWeightConsolidation(brain.parameters(), ewcLambda) : null;
    }

    private boolean grows() {
        return protection == Protection.GROW_FROZEN || protection == Protection.GROW_EWC;
    }

    /** @return the underlying brain */
    public Brain brain() {
        return brain;
    }

    /** @return the tasks added so far */
    public int tasks() {
        return heads.size();
    }

    /**
     * Starts a new task: wakes its region (growing modes) and adds its head.
     *
     * @param classes the task's classes
     * @return the task's index
     */
    public int addTask(int classes) {
        int t = heads.size();
        if (grows()) {
            if (t >= maxTasks) {
                throw new IllegalStateException("all " + maxTasks + " regions are in use");
            }
            brain.addNeurons("task" + t, regionSize, random);
        }
        int width = grows() ? (t + 1) * regionSize : regionSize;
        float std = (float) Math.sqrt(1.0 / width);
        heads.add(new Var[] {Var.param(Tensor.normal(Shape.of(width, classes), 0, std, random)),
            Var.param(Tensor.zeros(Shape.of(1, classes)))});
        headInputs.add(width);
        return t;
    }

    /**
     * @param task a task
     * @param x inputs
     * @return the task head's logits
     */
    public Var logits(int task, Var x) {
        Var features = brain.forward(x, false);
        int width = headInputs.get(task);
        if (features.shape().dim(1) > width) {
            features = features.slice(1, 0, width);
        }
        Var[] h = heads.get(task);
        return features.matmul(h[0]).add(h[1]);
    }

    /** The parameters the current task may change. */
    private List<Var> trainable(int task) {
        List<Var> out = new ArrayList<>();
        List<Var> all = brain.parameters();
        if (protection == Protection.GROW_FROZEN) {
            List<Brain.ProjectionStats> projections = brain.projections();
            String region = "task" + task;
            for (int j = 0; j < projections.size(); j++) {
                if (projections.get(j).to().equals(region)) {
                    out.add(all.get(j));
                }
            }
            out.add(all.get(projections.size() + task));   // the region's bias
        } else {
            out.addAll(all);
        }
        out.addAll(List.of(heads.get(task)));
        return out;
    }

    /**
     * Trains the current task.
     *
     * @param task the task (normally the newest)
     * @param x inputs
     * @param ids class ids within the task
     * @param epochs passes over the data
     * @param batch the batch size
     * @param optimizer a fresh optimizer for this task
     * @return the last epoch's mean loss
     */
    public float train(int task, Tensor x, Tensor ids, int epochs, int batch, Optimizer optimizer) {
        List<Var> params = trainable(task);
        int n = x.shape().dim(0);
        float last = 0;
        for (int e = 0; e < epochs; e++) {
            double sum = 0;
            int count = 0;
            for (int s = 0; s < n; s += batch) {
                int end = Math.min(n, s + batch);
                Var loss = Functional.crossEntropy(logits(task, Var.of(x.slice(0, s, end))), ids.slice(0, s, end));
                if (ewc != null) {
                    loss = loss.add(ewc.penalty());
                }
                loss.backward();
                sum += loss.value().item();
                count++;
                optimizer.step(params);
                for (Var p : brain.parameters()) {
                    p.zeroGrad();
                }
            }
            last = (float) (sum / count);
        }
        return last;
    }

    /**
     * Ends a task: for the EWC modes, measures the Fisher information on its data.
     *
     * @param task the task
     * @param x some of its inputs
     * @param ids their class ids
     */
    public void finishTask(int task, Tensor x, Tensor ids) {
        if (ewc == null) {
            return;
        }
        List<Supplier<Var>> batches = new ArrayList<>();
        int n = x.shape().dim(0);
        for (int s = 0; s < n; s += 8) {
            int from = s;
            int to = Math.min(n, s + 8);
            batches.add(() -> Functional.crossEntropy(logits(task, Var.of(x.slice(0, from, to))), ids.slice(0, from, to)));
        }
        ewc.consolidate(batches);
        for (Var[] h : heads) {
            h[0].zeroGrad();
            h[1].zeroGrad();
        }
    }

    /**
     * @param task a task
     * @param x inputs
     * @param ids class ids
     * @return the accuracy of the task's head
     */
    public double accuracy(int task, Tensor x, Tensor ids) {
        return LearningMath.accuracy(logits(task, Var.of(x)).value(), ids);
    }
}
