package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ReservoirReplay;
import io.github.zzuegg.brainiac.experimental.train.ContinualBrain;
import io.github.zzuegg.brainiac.experimental.train.DistillationReplay;
import io.github.zzuegg.brainiac.experimental.train.ElasticWeightConsolidation;
import io.github.zzuegg.brainiac.experimental.train.ForgettingTracker;
import io.github.zzuegg.brainiac.experimental.train.GradientEpisodicMemory;
import io.github.zzuegg.brainiac.experimental.train.ParameterIsolation;
import io.github.zzuegg.brainiac.experimental.train.SynapticIntelligence;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Class-incremental "split digits": the ten {@link ToyTasks#digits} classes arrive as five
 * tasks of two classes each (the split-MNIST protocol), learned one after another by one
 * network with one ten-way head. Unlike {@link ContinualLearningExperiment}'s permuted
 * digits (domain-incremental), the model must also tell apart classes it never saw
 * together, which is where plain regularizers such as EWC and SI fail.
 *
 * <p>Two accuracies are tracked after every task: <b>class-incremental</b> (argmax over every
 * class seen so far, no task identity) and <b>task-incremental</b> (argmax over the two
 * classes of the task, given its identity). Parameter-isolation methods (PackNet, HAT,
 * {@code GROW_FROZEN}) need the task identity and report only the latter.
 */
@SuppressWarnings("exports")
public final class ClassIncrementalDigits {

    /** The continual learners. */
    public enum Method {
        /** Plain sequential fine-tuning, the lower bound. */
        FINETUNE,
        /** Elastic weight consolidation. */
        EWC,
        /** Synaptic intelligence. */
        SI,
        /** Reservoir replay of raw samples. */
        REPLAY,
        /** Learning without forgetting. */
        LWF,
        /** iCaRL: herded exemplars, distillation, nearest mean of exemplars. */
        ICARL,
        /** Dark experience replay with labels. */
        DER_PLUS_PLUS,
        /** Averaged gradient episodic memory. */
        AGEM,
        /** Gradient episodic memory (one constraint per task). */
        GEM,
        /** PackNet. */
        PACKNET,
        /** Hard attention to the task. */
        HAT,
        /** Progressive networks: a new frozen-off region per task ({@link ContinualBrain}). */
        GROW_FROZEN
    }

    /**
     * One run.
     *
     * @param classIncremental class-incremental accuracies, or {@code null} for methods that
     *     need the task identity
     * @param taskIncremental task-incremental accuracies
     * @param millisPerStep mean wall time of one training step
     * @param capacityPerTask the mean share of the network each task claimed (isolation
     *     methods), or the memory in samples divided by the training-set size (replay)
     */
    public record Result(ForgettingTracker classIncremental, ForgettingTracker taskIncremental, double millisPerStep,
            double capacityPerTask) {
    }

    /**
     * One task.
     *
     * @param train the training split
     * @param test the test split
     * @param classes the task's classes
     */
    public record Task(ToyTasks.Data train, ToyTasks.Data test, int[] classes) {
    }

    private static final int BATCH = 32;

    private ClassIncrementalDigits() {
    }

    /**
     * The five two-class tasks.
     *
     * @param perClass training samples per class
     * @param seed the seed
     * @return the tasks
     */
    public static List<Task> split(int perClass, long seed) {
        ToyTasks.Data train = ToyTasks.digits(10, perClass, 8, 0.05, seed);
        ToyTasks.Data test = ToyTasks.digits(10, 20, 8, 0.05, seed, seed + 1000);
        List<Task> out = new ArrayList<>();
        for (int t = 0; t < 5; t++) {
            int[] c = {2 * t, 2 * t + 1};
            out.add(new Task(train.onlyClasses(c), test.onlyClasses(c), c));
        }
        return out;
    }

    private interface Learner {
        /** One training step on a batch of the current task. */
        void step(int task, Tensor x, Tensor ids, int batch, int batches);

        /** Called after all epochs of a task. */
        void finish(int task, Task data);

        /** Logits {@code [n, 10]} as task {@code task} would produce them. */
        Tensor logits(int task, Tensor x);

        /** Class predictions without task identity, or null if the method cannot. */
        default int[] classify(Tensor x, int seenClasses) {
            return null;
        }
    }

    /**
     * Trains one method on the tasks in order.
     *
     * @param method the method
     * @param tasks the tasks
     * @param epochs epochs per task
     * @param memory the replay / exemplar budget in samples (for GEM split over the tasks)
     * @param seed the seed
     * @return the result
     */
    public static Result run(Method method, List<Task> tasks, int epochs, int memory, long seed) {
        if (method == Method.GROW_FROZEN) {
            return progressive(tasks, epochs, seed);
        }
        Model model = Model.sequential().input(64).dense(64, Activation.RELU).dense(64, Activation.RELU)
                .dense(10, Activation.IDENTITY).seed(seed).build();
        Learner learner = learner(method, model, tasks, memory, seed);
        ForgettingTracker cil = new ForgettingTracker();
        ForgettingTracker til = new ForgettingTracker();
        SplittableRandom r = new SplittableRandom(seed);
        long steps = 0;
        long nanos = 0;
        boolean classIl = true;
        for (int t = 0; t < tasks.size(); t++) {
            Task task = tasks.get(t);
            int n = task.train().size();
            int batches = (n + BATCH - 1) / BATCH;
            int[] order = new int[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            for (int e = 0; e < epochs; e++) {
                shuffle(order, r);
                for (int b = 0; b < batches; b++) {
                    int from = b * BATCH;
                    int to = Math.min(n, from + BATCH);
                    float[][] x = new float[to - from][];
                    float[] y = new float[to - from];
                    for (int i = from; i < to; i++) {
                        x[i - from] = task.train().x()[order[i]];
                        y[i - from] = task.train().y()[order[i]];
                    }
                    long t0 = System.nanoTime();
                    learner.step(t, Tensor.of(x), Tensor.of(y), b, batches);
                    nanos += System.nanoTime() - t0;
                    steps++;
                }
            }
            learner.finish(t, task);
            double[] c = new double[t + 1];
            double[] k = new double[t + 1];
            for (int j = 0; j <= t; j++) {
                Task old = tasks.get(j);
                int[] pred = learner.classify(old.test().inputs(), 2 * (t + 1));
                if (pred == null) {
                    classIl = false;
                } else {
                    c[j] = accuracy(pred, old.test().y());
                }
                k[j] = taskAccuracy(learner.logits(j, old.test().inputs()), old.test().y(), old.classes());
            }
            cil.record(c);
            til.record(k);
        }
        double capacity = switch (method) {
            case REPLAY, ICARL, DER_PLUS_PLUS, AGEM, GEM -> memory / (double) (tasks.get(0).train().size() * 5);
            default -> learner instanceof Isolated iso ? iso.capacity() : 0;
        };
        return new Result(classIl ? cil : null, til, nanos / 1e6 / steps, capacity);
    }

    private interface Isolated {
        double capacity();
    }

    private static Learner learner(Method method, Model model, List<Task> tasks, int memory, long seed) {
        Optimizer adam = Optimizer.adam(1e-3f);
        List<Var> params = model.parameters();
        return switch (method) {
            case FINETUNE -> plain(model, adam, () -> null, (t, d) -> { });
            case EWC -> {
                ElasticWeightConsolidation ewc = new ElasticWeightConsolidation(params, 500f);
                yield plain(model, adam, ewc::penalty, (t, d) -> ewc.consolidate(batches(model, d.train())));
            }
            case SI -> {
                SynapticIntelligence si = new SynapticIntelligence(params, 1f, 0.1f);
                yield plain(model, si.track(adam), si::penalty, (t, d) -> si.consolidate());
            }
            case REPLAY -> replay(model, adam, memory, seed);
            case LWF, ICARL, DER_PLUS_PLUS -> distillation(method, model, adam, memory, seed);
            case AGEM, GEM -> {
                GradientEpisodicMemory gem = new GradientEpisodicMemory(params,
                        (x, ids) -> Functional.crossEntropy(model.forward(Var.of(x)), ids),
                        memory / (tasks.size() - 1), method == Method.AGEM, BATCH, seed);
                yield plain(model, gem.wrap(adam), () -> null,
                        (t, d) -> gem.remember(d.train().inputs(), d.train().ids()));
            }
            case PACKNET -> packNet(model, adam, tasks);
            case HAT -> hat(tasks, seed);
            case GROW_FROZEN -> throw new IllegalArgumentException("handled separately");
        };
    }

    private interface Finish {
        void finish(int task, Task data);
    }

    /** A learner whose loss is cross-entropy plus an optional penalty. */
    private static Learner plain(Model model, Optimizer opt, Supplier<Var> penalty, Finish finish) {
        return new Learner() {
            @Override
            public void step(int task, Tensor x, Tensor ids, int batch, int batches) {
                Var loss = Functional.crossEntropy(model.forward(Var.of(x), true), ids);
                Var p = penalty.get();
                (p == null ? loss : loss.add(p)).backward();
                opt.step(model.parameters());
            }

            @Override
            public void finish(int task, Task data) {
                finish.finish(task, data);
            }

            @Override
            public Tensor logits(int task, Tensor x) {
                return model.forward(Var.of(x)).value();
            }

            @Override
            public int[] classify(Tensor x, int seenClasses) {
                return argmax(model.forward(Var.of(x)).value(), seenClasses);
            }
        };
    }

    private static List<Supplier<Var>> batches(Model model, ToyTasks.Data d) {
        List<Supplier<Var>> out = new ArrayList<>();
        for (int from = 0; from < d.size(); from += BATCH) {
            ToyTasks.Data part = d.slice(from, Math.min(d.size(), from + BATCH));
            out.add(() -> Functional.crossEntropy(model.forward(Var.of(part.inputs())), part.ids()));
        }
        return out;
    }

    private static Learner replay(Model model, Optimizer opt, int memory, long seed) {
        ReservoirReplay buffer = new ReservoirReplay(memory, seed);
        Learner base = plain(model, opt, () -> null, (t, d) -> { });
        return new Learner() {
            @Override
            public void step(int task, Tensor x, Tensor ids, int batch, int batches) {
                Tensor[] mixed = buffer.mix(x, ids, x.shape().dim(0));
                buffer.add(x, ids);
                base.step(task, mixed[0], mixed[1], batch, batches);
            }

            @Override
            public void finish(int task, Task data) {
            }

            @Override
            public Tensor logits(int task, Tensor x) {
                return base.logits(task, x);
            }

            @Override
            public int[] classify(Tensor x, int seenClasses) {
                return base.classify(x, seenClasses);
            }
        };
    }

    private static Learner distillation(Method method, Model model, Optimizer opt, int memory, long seed) {
        DistillationReplay.Kind kind = switch (method) {
            case LWF -> DistillationReplay.Kind.LWF;
            case ICARL -> DistillationReplay.Kind.ICARL;
            default -> DistillationReplay.Kind.DER_PLUS_PLUS;
        };
        DistillationReplay dr = new DistillationReplay(kind, model, memory, kind == DistillationReplay.Kind.DER_PLUS_PLUS
                ? 0.3f : 1f, 0.5f, 2f, seed);
        return new Learner() {
            @Override
            public void step(int task, Tensor x, Tensor ids, int batch, int batches) {
                dr.loss(x, ids).backward();
                opt.step(model.parameters());
            }

            @Override
            public void finish(int task, Task data) {
                dr.finishTask(data.train().inputs(), data.train().ids());
            }

            @Override
            public Tensor logits(int task, Tensor x) {
                return model.forward(Var.of(x)).value();
            }

            @Override
            public int[] classify(Tensor x, int seenClasses) {
                return kind == DistillationReplay.Kind.ICARL ? dr.classify(x)
                        : argmax(model.forward(Var.of(x)).value(), seenClasses);
            }
        };
    }

    private abstract static class IsolatedLearner implements Learner, Isolated {
    }

    private static Learner packNet(Model model, Optimizer adam, List<Task> tasks) {
        ParameterIsolation.PackNet pack = ParameterIsolation.packNet(model.parameters(), 0.5);
        Optimizer opt = pack.wrap(adam);
        return new IsolatedLearner() {
            private int current = -1;

            @Override
            public void step(int task, Tensor x, Tensor ids, int batch, int batches) {
                if (task != current) {
                    pack.startTask(task);
                    current = task;
                }
                Functional.crossEntropy(model.forward(Var.of(x), true), ids).backward();
                opt.step(model.parameters());
            }

            @Override
            public void finish(int task, Task data) {
                pack.prune(1.0 / (tasks.size() - task));
                // a short retraining of the task's surviving weights
                ToyTasks.Data d = data.train();
                for (int e = 0; e < 5; e++) {
                    for (int from = 0; from < d.size(); from += BATCH) {
                        ToyTasks.Data part = d.slice(from, Math.min(d.size(), from + BATCH));
                        Functional.crossEntropy(model.forward(Var.of(part.inputs()), true), part.ids()).backward();
                        opt.step(model.parameters());
                    }
                }
            }

            @Override
            public Tensor logits(int task, Tensor x) {
                pack.use(task);
                Tensor out = model.forward(Var.of(x)).value();
                pack.restore();
                return out;
            }

            @Override
            public double capacity() {
                double sum = 0;
                for (int t = 0; t < tasks.size(); t++) {
                    sum += pack.capacity(t);
                }
                return sum / tasks.size();
            }
        };
    }

    private static Learner hat(List<Task> tasks, long seed) {
        ParameterIsolation.Hat hat = ParameterIsolation.hat(64, new int[] {64, 64}, 10, tasks.size(),
                new SplittableRandom(seed));
        Optimizer opt = hat.wrap(Optimizer.adam(1e-3f), Optimizer.adam(0.05f));
        return new IsolatedLearner() {
            @Override
            public void step(int task, Tensor x, Tensor ids, int batch, int batches) {
                float s = ParameterIsolation.Hat.slope(batch, batches);
                Functional.crossEntropy(hat.forward(Var.of(x), task, s), ids).add(hat.regularizer(task, s)).backward();
                opt.step(hat.parameters());
            }

            @Override
            public void finish(int task, Task data) {
                hat.finishTask(task, data.classes());
            }

            @Override
            public Tensor logits(int task, Tensor x) {
                return hat.forward(Var.of(x), task, ParameterIsolation.Hat.S_MAX).value();
            }

            @Override
            public double capacity() {
                double sum = 0;
                for (int t = 0; t < tasks.size(); t++) {
                    sum += hat.capacity(t);
                }
                return sum / tasks.size();
            }
        };
    }

    /** Progressive networks via {@link ContinualBrain}: task-local labels, one region per task. */
    private static Result progressive(List<Task> tasks, int epochs, long seed) {
        ContinualBrain brain = new ContinualBrain(64, 32, tasks.size(), 0.3, ContinualBrain.Protection.GROW_FROZEN,
                500f, seed);
        ForgettingTracker til = new ForgettingTracker();
        long t0 = System.nanoTime();
        long steps = 0;
        for (int t = 0; t < tasks.size(); t++) {
            Task task = tasks.get(t);
            int id = brain.addTask(2);
            Tensor x = task.train().inputs();
            Tensor y = local(task.train().y(), task.classes());
            brain.train(id, x, y, epochs, BATCH, Optimizer.adam(0.01f));
            steps += (long) epochs * ((task.train().size() + BATCH - 1) / BATCH);
            brain.finishTask(id, x.slice(0, 0, Math.min(64, task.train().size())),
                    y.slice(0, 0, Math.min(64, task.train().size())));
            double[] acc = new double[t + 1];
            for (int j = 0; j <= t; j++) {
                Task old = tasks.get(j);
                acc[j] = brain.accuracy(j, old.test().inputs(), local(old.test().y(), old.classes()));
            }
            til.record(acc);
        }
        return new Result(null, til, (System.nanoTime() - t0) / 1e6 / steps, 1.0 / tasks.size());
    }

    private static Tensor local(int[] y, int[] classes) {
        float[] out = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = y[i] == classes[0] ? 0 : 1;
        }
        return Tensor.of(out);
    }

    private static int[] argmax(Tensor logits, int classes) {
        int k = logits.shape().dim(1);
        float[] l = logits.toArray();
        int[] out = new int[logits.shape().dim(0)];
        for (int i = 0; i < out.length; i++) {
            for (int c = 1; c < classes; c++) {
                if (l[i * k + c] > l[i * k + out[i]]) {
                    out[i] = c;
                }
            }
        }
        return out;
    }

    private static double taskAccuracy(Tensor logits, int[] y, int[] classes) {
        int k = logits.shape().dim(1);
        float[] l = logits.toArray();
        int ok = 0;
        for (int i = 0; i < y.length; i++) {
            int pred = l[i * k + classes[0]] >= l[i * k + classes[1]] ? classes[0] : classes[1];
            ok += pred == y[i] ? 1 : 0;
        }
        return ok / (double) y.length;
    }

    private static double accuracy(int[] pred, int[] y) {
        int ok = 0;
        for (int i = 0; i < y.length; i++) {
            ok += pred[i] == y[i] ? 1 : 0;
        }
        return ok / (double) y.length;
    }

    private static void shuffle(int[] a, SplittableRandom r) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }
}
