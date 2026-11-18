package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Multi-task ranking with expert gates, as a {@link Layer} from {@code [batch, inputs]} features to
 * one logit per task, {@code [batch, tasks]}. Ranking systems predict several labels at once
 * (click, like, convert); a shared bottom suffers <em>negative transfer</em> when the tasks
 * conflict.
 * <ul>
 *   <li>{@link Kind#SHARED_BOTTOM}: one MLP bottom shared by every task tower (the baseline);</li>
 *   <li>{@link Kind#MMOE} (Ma et al., KDD 2018, <a href="https://doi.org/10.1145/3219819.3220007">
 *       doi:10.1145/3219819.3220007</a>): a pool of expert MLPs, and per task a softmax gate over
 *       them, so each task mixes the experts it needs;</li>
 *   <li>{@link Kind#PLE} (Tang et al., RecSys 2020, <a href="https://doi.org/10.1145/3383313.3412236">
 *       doi:10.1145/3383313.3412236</a>): progressive layered extraction. Each extraction layer has
 *       task-specific experts and shared experts; a task's gate mixes its own and the shared
 *       experts, and a shared gate mixes all experts for the next layer. Separating the experts
 *       removes MMoE's "seesaw" where one task gains only at another's expense.</li>
 * </ul>
 * Every task has a tower (one hidden ReLU layer) on top. {@link #taskRepresentation} exposes the
 * tower inputs for heads such as {@link EntireSpaceModel}.
 */
public final class MultiGateExperts implements Layer {

    /** The architecture. */
    public enum Kind {
        /** Shared-bottom MLP. */
        SHARED_BOTTOM,
        /** Multi-gate mixture of experts. */
        MMOE,
        /** Progressive layered extraction. */
        PLE
    }

    /** One two-layer ReLU expert. */
    private record Expert(Var w1, Var b1, Var w2, Var b2) {
        Var apply(Var x) {
            return x.matmul(w1).add(b1).relu().matmul(w2).add(b2).relu();
        }

        List<Var> parameters() {
            return List.of(w1, b1, w2, b2);
        }
    }

    /** A softmax gate. */
    private record Gate(Var w, Var b) {
        Var apply(Var x, List<Var> experts) {
            int n = x.shape().dim(0);
            int e = experts.size();
            int d = experts.get(0).shape().dim(1);
            Var logits = x.matmul(w).add(b);
            Var g = logits.sub(logits.logsumexp(1, true)).exp().reshape(n, e, 1);
            List<Var> stacked = new ArrayList<>();
            for (Var v : experts) {
                stacked.add(v.reshape(n, 1, d));
            }
            return Var.concat(1, stacked.toArray(Var[]::new)).mul(g).sum(1, false);
        }
    }

    private final Kind kind;
    private final int tasks;
    private final int inputs;
    private final int expertDim;
    private final List<List<Expert>> taskExperts = new ArrayList<>();
    private final List<Expert> sharedExperts = new ArrayList<>();
    private final List<List<Gate>> taskGates = new ArrayList<>();
    private final List<Gate> sharedGates = new ArrayList<>();
    private final int layers;
    private final List<Var> towers = new ArrayList<>();

    /**
     * Creates the layer.
     *
     * @param kind the architecture
     * @param inputs input features
     * @param tasks tasks (output logits)
     * @param experts for MMoE the pool size; for PLE the experts per task and the shared experts
     *     per layer; ignored by the shared bottom
     * @param hidden the experts' (and bottom's) hidden width
     * @param expertDim the experts' output width
     * @param layers PLE extraction layers (1 or more); ignored otherwise
     * @param random the initializer
     */
    public MultiGateExperts(Kind kind, int inputs, int tasks, int experts, int hidden, int expertDim, int layers,
            SplittableRandom random) {
        this.kind = kind;
        this.tasks = tasks;
        this.inputs = inputs;
        this.expertDim = expertDim;
        this.layers = kind == Kind.PLE ? layers : 1;
        switch (kind) {
            case SHARED_BOTTOM -> sharedExperts.add(expert(inputs, hidden, expertDim, random));
            case MMOE -> {
                for (int e = 0; e < experts; e++) {
                    sharedExperts.add(expert(inputs, hidden, expertDim, random));
                }
                List<Gate> gates = new ArrayList<>();
                for (int t = 0; t < tasks; t++) {
                    gates.add(gate(inputs, experts, random));
                }
                taskGates.add(gates);
            }
            case PLE -> {
                for (int l = 0; l < layers; l++) {
                    int in = l == 0 ? inputs : expertDim;
                    List<Gate> gates = new ArrayList<>();
                    for (int t = 0; t < tasks; t++) {
                        List<Expert> own = new ArrayList<>();
                        for (int e = 0; e < experts; e++) {
                            own.add(expert(in, hidden, expertDim, random));
                        }
                        taskExperts.add(own);
                        gates.add(gate(in, 2 * experts, random));
                    }
                    taskGates.add(gates);
                    for (int e = 0; e < experts; e++) {
                        sharedExperts.add(expert(in, hidden, expertDim, random));
                    }
                    if (l + 1 < layers) {
                        sharedGates.add(gate(in, experts * (tasks + 1), random));
                    }
                }
            }
            default -> throw new IllegalArgumentException(kind.toString());
        }
        for (int t = 0; t < tasks; t++) {
            towers.add(RecsysOps.matrix(random, expertDim, hidden));
            towers.add(RecsysOps.zeros(hidden));
            towers.add(RecsysOps.matrix(random, hidden, 1));
            towers.add(RecsysOps.zeros(1));
        }
    }

    /**
     * @param kind the architecture
     * @param tasks tasks
     * @param experts experts (see the constructor)
     * @param hidden hidden width
     * @param expertDim expert output width
     * @param layers PLE layers
     * @return a spec for model builders
     */
    public static LayerSpec spec(Kind kind, int tasks, int experts, int hidden, int expertDim, int layers) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(tasks);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new MultiGateExperts(kind, input.dim(-1), tasks, experts, hidden, expertDim, layers, random);
            }
        };
    }

    private static Expert expert(int in, int hidden, int out, SplittableRandom r) {
        return new Expert(RecsysOps.matrix(r, in, hidden), RecsysOps.zeros(hidden), RecsysOps.matrix(r, hidden, out),
                RecsysOps.zeros(out));
    }

    private static Gate gate(int in, int experts, SplittableRandom r) {
        return new Gate(RecsysOps.normal(r, 0.01f, in, experts), RecsysOps.zeros(experts));
    }

    /**
     * The inputs of the task towers.
     *
     * @param input {@code [batch, inputs]}
     * @return one {@code [batch, expertDim]} representation per task
     */
    public List<Var> taskRepresentation(Var input) {
        List<Var> out = new ArrayList<>();
        switch (kind) {
            case SHARED_BOTTOM -> {
                Var h = sharedExperts.get(0).apply(input);
                for (int t = 0; t < tasks; t++) {
                    out.add(h);
                }
            }
            case MMOE -> {
                List<Var> e = new ArrayList<>();
                for (Expert x : sharedExperts) {
                    e.add(x.apply(input));
                }
                for (int t = 0; t < tasks; t++) {
                    out.add(taskGates.get(0).get(t).apply(input, e));
                }
            }
            case PLE -> {
                List<Var> taskIn = new ArrayList<>();
                for (int t = 0; t < tasks; t++) {
                    taskIn.add(input);
                }
                Var sharedIn = input;
                int per = sharedExperts.size() / layers;
                for (int l = 0; l < layers; l++) {
                    List<Var> shared = new ArrayList<>();
                    for (int e = 0; e < per; e++) {
                        shared.add(sharedExperts.get(l * per + e).apply(sharedIn));
                    }
                    List<Var> all = new ArrayList<>(shared);
                    List<Var> next = new ArrayList<>();
                    for (int t = 0; t < tasks; t++) {
                        List<Var> mine = new ArrayList<>();
                        for (Expert x : taskExperts.get(l * tasks + t)) {
                            mine.add(x.apply(taskIn.get(t)));
                        }
                        all.addAll(mine);
                        List<Var> pool = new ArrayList<>(mine);
                        pool.addAll(shared);
                        next.add(taskGates.get(l).get(t).apply(taskIn.get(t), pool));
                    }
                    if (l + 1 < layers) {
                        sharedIn = sharedGates.get(l).apply(sharedIn, all);
                    }
                    taskIn = next;
                }
                out = taskIn;
            }
            default -> throw new IllegalStateException();
        }
        return out;
    }

    /**
     * Applies one task's tower.
     *
     * @param task the task
     * @param representation its representation from {@link #taskRepresentation}
     * @return {@code [batch, 1]} logits
     */
    public Var tower(int task, Var representation) {
        int o = 4 * task;
        return representation.matmul(towers.get(o)).add(towers.get(o + 1)).relu().matmul(towers.get(o + 2))
                .add(towers.get(o + 3));
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().dim(-1) != inputs) {
            throw new IllegalArgumentException("expected " + inputs + " features, got " + input.shape());
        }
        List<Var> reps = taskRepresentation(input);
        List<Var> logits = new ArrayList<>();
        for (int t = 0; t < tasks; t++) {
            logits.add(tower(t, reps.get(t)));
        }
        return Var.concat(1, logits.toArray(Var[]::new));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Expert e : sharedExperts) {
            p.addAll(e.parameters());
        }
        for (List<Expert> list : taskExperts) {
            for (Expert e : list) {
                p.addAll(e.parameters());
            }
        }
        for (List<Gate> list : taskGates) {
            for (Gate g : list) {
                p.add(g.w());
                p.add(g.b());
            }
        }
        for (Gate g : sharedGates) {
            p.add(g.w());
            p.add(g.b());
        }
        p.addAll(towers);
        return p;
    }

    /** @return the number of tasks */
    public int tasks() {
        return tasks;
    }

    /** @return the expert output width */
    public int expertDim() {
        return expertDim;
    }

    @Override
    public int outputSize(int inputSize) {
        return tasks;
    }

    @Override
    public String toString() {
        return "MultiGateExperts(" + kind + ", " + tasks + " tasks)";
    }
}
