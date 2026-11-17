package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * AutoML-Zero, a light version: evolves whole learning algorithms from scratch as tiny
 * programs over a small instruction set. A program has three components, run on a memory
 * of scalars and vectors:
 *
 * <ul>
 *   <li>{@code setup}, run once per task (say, initialise weights and a learning rate);</li>
 *   <li>{@code predict}, run per example with the features in vector 0, leaving the
 *       prediction in scalar 1;</li>
 *   <li>{@code learn}, run after {@code predict} on training examples with the label in
 *       scalar 0.</li>
 * </ul>
 *
 * <p>Fitness is the validation loss after one pass over a task's training examples,
 * averaged over several random tasks (by default random linear regressions), so only
 * programs that actually <em>learn</em> score well. Search is regularized (aging)
 * evolution: tournament selection, and the oldest individual dies each cycle.
 *
 * <p>Real et al. 2020, "AutoML-Zero: Evolving Machine Learning Algorithms From Scratch",
 * <a href="https://arxiv.org/abs/2003.03384">arXiv:2003.03384</a>.
 */
public final class AutoMlZero {

    /** The instruction set. */
    public enum Op {
        /** {@code s[out] = s[a] + s[b]} */
        S_ADD,
        /** {@code s[out] = s[a] - s[b]} */
        S_SUB,
        /** {@code s[out] = s[a] * s[b]} */
        S_MUL,
        /** {@code s[out] = s[a] / s[b]} (0 when dividing by 0) */
        S_DIV,
        /** {@code s[out] = constant} */
        S_CONST,
        /** {@code s[out] = |s[a]|} */
        S_ABS,
        /** {@code s[out] = tanh(s[a])} */
        S_TANH,
        /** {@code v[out] = v[a] + v[b]} */
        V_ADD,
        /** {@code v[out] = v[a] - v[b]} */
        V_SUB,
        /** {@code v[out] = v[a] * v[b]} element-wise */
        V_MUL,
        /** {@code v[out] = s[a] * v[b]} */
        SV_MUL,
        /** {@code s[out] = dot(v[a], v[b])} */
        V_DOT,
        /** {@code s[out] = mean(v[a])} */
        V_MEAN,
        /** {@code v[out] = uniform(-c, c)} */
        V_RANDOM
    }

    /**
     * One instruction.
     *
     * @param op the operation
     * @param out the output address
     * @param a the first input address
     * @param b the second input address
     * @param constant the constant, for {@code S_CONST} and {@code V_RANDOM}
     */
    public record Instruction(Op op, int out, int a, int b, double constant) {
        @Override
        public String toString() {
            return switch (op) {
                case S_ADD -> "s" + out + " = s" + a + " + s" + b;
                case S_SUB -> "s" + out + " = s" + a + " - s" + b;
                case S_MUL -> "s" + out + " = s" + a + " * s" + b;
                case S_DIV -> "s" + out + " = s" + a + " / s" + b;
                case S_CONST -> String.format("s%d = %.4g", out, constant);
                case S_ABS -> "s" + out + " = |s" + a + "|";
                case S_TANH -> "s" + out + " = tanh(s" + a + ")";
                case V_ADD -> "v" + out + " = v" + a + " + v" + b;
                case V_SUB -> "v" + out + " = v" + a + " - v" + b;
                case V_MUL -> "v" + out + " = v" + a + " * v" + b;
                case SV_MUL -> "v" + out + " = s" + a + " * v" + b;
                case V_DOT -> "s" + out + " = dot(v" + a + ", v" + b + ")";
                case V_MEAN -> "s" + out + " = mean(v" + a + ")";
                case V_RANDOM -> String.format("v%d = uniform(+-%.3g)", out, constant);
            };
        }
    }

    /**
     * A learning algorithm.
     *
     * @param setup run once per task
     * @param predict run per example, features in v0, prediction to s1
     * @param learn run after predict on training examples, label in s0
     */
    public record Program(List<Instruction> setup, List<Instruction> predict, List<Instruction> learn) {
        /** Copies the lists. */
        public Program {
            setup = List.copyOf(setup);
            predict = List.copyOf(predict);
            learn = List.copyOf(learn);
        }

        /** @return an empty program, which predicts 0 */
        public static Program empty() {
            return new Program(List.of(), List.of(), List.of());
        }

        @Override
        public String toString() {
            return "setup:\n  " + String.join("\n  ", setup.stream().map(Object::toString).toList())
                    + "\npredict:\n  " + String.join("\n  ", predict.stream().map(Object::toString).toList())
                    + "\nlearn:\n  " + String.join("\n  ", learn.stream().map(Object::toString).toList());
        }
    }

    /**
     * A supervised task: training and validation examples.
     *
     * @param trainX training features
     * @param trainY training labels
     * @param validX validation features
     * @param validY validation labels
     * @param classification whether labels are -1/+1 classes scored by the sign of the
     *     prediction (fitness: accuracy minus 1) rather than values scored by squared error
     */
    public record Task(double[][] trainX, double[] trainY, double[][] validX, double[] validY,
            boolean classification) {
    }

    private static final int MAX_LENGTH = 8;

    private int scalars = 6;
    private int vectors = 4;
    private Op[] ops = Op.values();

    private final int features;
    private int population = 100;
    private int tournament = 10;
    private long seed = 1;
    private Program best = Program.empty();
    private double bestFitness = Double.NEGATIVE_INFINITY;
    private int evaluations;

    private AutoMlZero(int features) {
        this.features = features;
    }

    /**
     * Sets up a search for tasks with a number of features.
     *
     * @param features the feature vector length
     * @return the search
     */
    public static AutoMlZero of(int features) {
        return new AutoMlZero(features);
    }

    /**
     * Sets the population and tournament sizes.
     *
     * @param size the population
     * @param tournamentSize the tournament
     * @return this
     */
    public AutoMlZero population(int size, int tournamentSize) {
        this.population = size;
        this.tournament = tournamentSize;
        return this;
    }

    /**
     * Restricts the search space, as the paper does for its smaller experiments.
     *
     * @param scalarCount scalar memory size (at least 2: label and prediction)
     * @param vectorCount vector memory size (at least 2: features and one more)
     * @param allowed the instructions mutation may use
     * @return this
     */
    public AutoMlZero searchSpace(int scalarCount, int vectorCount, Op... allowed) {
        if (scalarCount < 2 || vectorCount < 2 || allowed.length == 0) {
            throw new IllegalArgumentException("need 2 scalars, 2 vectors and some instructions");
        }
        this.scalars = scalarCount;
        this.vectors = vectorCount;
        this.ops = allowed.clone();
        return this;
    }

    /**
     * Sets the seed.
     *
     * @param s the seed
     * @return this
     */
    public AutoMlZero seed(long s) {
        this.seed = s;
        return this;
    }

    /** @return the best program found */
    public Program best() {
        return best;
    }

    /** @return its fitness (minus the mean normalised validation loss) */
    public double bestFitness() {
        return bestFitness;
    }

    /** @return the number of programs evaluated */
    public int evaluations() {
        return evaluations;
    }

    /**
     * Random linear regression tasks: {@code y = w . x} with Gaussian {@code x} and a
     * random {@code w} per task.
     *
     * @param count the number of tasks
     * @param features the feature count
     * @param train training examples per task
     * @param valid validation examples per task
     * @param seed the seed
     * @return the tasks
     */
    public static List<Task> linearTasks(int count, int features, int train, int valid, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Task> out = new ArrayList<>();
        for (int t = 0; t < count; t++) {
            double[] w = new double[features];
            for (int i = 0; i < features; i++) {
                w[i] = r.nextGaussian();
            }
            double[][] tx = new double[train][];
            double[] ty = new double[train];
            double[][] vx = new double[valid][];
            double[] vy = new double[valid];
            for (int i = 0; i < train + valid; i++) {
                double[] x = new double[features];
                double y = 0;
                for (int j = 0; j < features; j++) {
                    x[j] = r.nextGaussian();
                    y += w[j] * x[j];
                }
                if (i < train) {
                    tx[i] = x;
                    ty[i] = y;
                } else {
                    vx[i - train] = x;
                    vy[i - train] = y;
                }
            }
            out.add(new Task(tx, ty, vx, vy, false));
        }
        return out;
    }

    /**
     * Random linearly separable binary classification tasks: label {@code sign(w . x)}
     * as -1 or +1.
     *
     * @param count the number of tasks
     * @param features the feature count
     * @param train training examples per task
     * @param valid validation examples per task
     * @param seed the seed
     * @return the tasks
     */
    public static List<Task> linearClassificationTasks(int count, int features, int train, int valid, long seed) {
        List<Task> out = new ArrayList<>();
        for (Task t : linearTasks(count, features, train, valid, seed)) {
            double[] ty = t.trainY().clone();
            double[] vy = t.validY().clone();
            for (int i = 0; i < ty.length; i++) {
                ty[i] = ty[i] >= 0 ? 1 : -1;
            }
            for (int i = 0; i < vy.length; i++) {
                vy[i] = vy[i] >= 0 ? 1 : -1;
            }
            out.add(new Task(t.trainX(), ty, t.validX(), vy, true));
        }
        return out;
    }

    /**
     * Scores a program: minus its validation loss, normalised by the label variance and
     * averaged over the tasks (0 is perfect, -1 is as good as predicting 0); for
     * classification tasks minus the validation error rate.
     *
     * @param program the program
     * @param tasks the tasks
     * @param seed the seed for random instructions
     * @return the fitness
     */
    public double fitness(Program program, List<Task> tasks, long seed) {
        evaluations++;
        double total = 0;
        SplittableRandom r = new SplittableRandom(seed);
        for (Task task : tasks) {
            double[] s = new double[Math.max(scalars, maxAddress(program, true) + 1)];
            double[][] v = new double[Math.max(vectors, maxAddress(program, false) + 1)][features];
            run(program.setup(), s, v, r);
            for (int i = 0; i < task.trainX().length; i++) {
                v[0] = task.trainX()[i].clone();
                run(program.predict(), s, v, r);
                s[0] = task.trainY()[i];
                run(program.learn(), s, v, r);
            }
            double loss = 0;
            double var = 0;
            for (int i = 0; i < task.validX().length; i++) {
                v[0] = task.validX()[i].clone();
                run(program.predict(), s, v, r);
                if (task.classification()) {
                    loss += s[1] * task.validY()[i] > 0 ? 0 : 1;
                    var += 1;
                } else {
                    double e = s[1] - task.validY()[i];
                    loss += e * e;
                    var += task.validY()[i] * task.validY()[i];
                }
            }
            double norm = loss / Math.max(var, 1e-12);
            total += Double.isFinite(norm) ? Math.min(norm, 10) : 10;
        }
        return -total / tasks.size();
    }

    private static void run(List<Instruction> code, double[] s, double[][] v, SplittableRandom r) {
        for (Instruction in : code) {
            switch (in.op()) {
                case S_ADD -> s[in.out()] = s[in.a()] + s[in.b()];
                case S_SUB -> s[in.out()] = s[in.a()] - s[in.b()];
                case S_MUL -> s[in.out()] = s[in.a()] * s[in.b()];
                case S_DIV -> s[in.out()] = s[in.b()] == 0 ? 0 : s[in.a()] / s[in.b()];
                case S_CONST -> s[in.out()] = in.constant();
                case S_ABS -> s[in.out()] = Math.abs(s[in.a()]);
                case S_TANH -> s[in.out()] = Math.tanh(s[in.a()]);
                case V_ADD, V_SUB, V_MUL -> {
                    double[] a = v[in.a()];
                    double[] b = v[in.b()];
                    double[] o = new double[a.length];
                    for (int i = 0; i < o.length; i++) {
                        o[i] = in.op() == Op.V_ADD ? a[i] + b[i] : in.op() == Op.V_SUB ? a[i] - b[i] : a[i] * b[i];
                    }
                    v[in.out()] = o;
                }
                case SV_MUL -> {
                    double c = s[in.a()];
                    double[] b = v[in.b()];
                    double[] o = new double[b.length];
                    for (int i = 0; i < o.length; i++) {
                        o[i] = c * b[i];
                    }
                    v[in.out()] = o;
                }
                case V_DOT -> {
                    double d = 0;
                    for (int i = 0; i < v[in.a()].length; i++) {
                        d += v[in.a()][i] * v[in.b()][i];
                    }
                    s[in.out()] = d;
                }
                case V_MEAN -> s[in.out()] = Arrays.stream(v[in.a()]).average().orElse(0);
                case V_RANDOM -> {
                    double[] o = new double[v[in.out()].length];
                    for (int i = 0; i < o.length; i++) {
                        o[i] = r.nextDouble(-1, 1) * in.constant();
                    }
                    v[in.out()] = o;
                }
                default -> throw new IllegalStateException();
            }
        }
    }

    /**
     * Runs regularized evolution.
     *
     * @param tasks the tasks fitness is measured on
     * @param cycles the number of children to evaluate
     * @return the best program
     */
    public Program evolve(List<Task> tasks, int cycles) {
        return evolve(cycle -> tasks, cycles);
    }

    /**
     * Runs regularized evolution with fresh tasks for every evaluation, which rewards
     * programs that learn rather than ones that fit particular tasks.
     *
     * @param tasks gives the tasks for an evaluation number
     * @param cycles the number of children to evaluate
     * @return the surviving program that scores best on a few further task sets
     */
    public Program evolve(java.util.function.LongFunction<List<Task>> tasks, int cycles) {
        SplittableRandom r = new SplittableRandom(seed);
        record Member(Program program, double fitness) {
        }
        ArrayDeque<Member> pop = new ArrayDeque<>();
        for (int i = 0; i < population; i++) {
            Program p = randomProgram(r);
            Member m = new Member(p, fitness(p, tasks.apply(i), seed + i));
            track(m.program(), m.fitness());
            pop.add(m);
        }
        for (int c = 0; c < cycles; c++) {
            List<Member> list = new ArrayList<>(pop);
            Member parent = null;
            for (int k = 0; k < tournament; k++) {
                Member m = list.get(r.nextInt(list.size()));
                if (parent == null || m.fitness() > parent.fitness()) {
                    parent = m;
                }
            }
            Program child = mutate(parent.program(), r);
            Member m = new Member(child, fitness(child, tasks.apply(population + c), seed + c));
            track(child, m.fitness());
            pop.add(m);
            pop.removeFirst();   // the oldest dies
        }
        // fitness was measured on random tasks: re-score the survivors on common ones
        bestFitness = Double.NEGATIVE_INFINITY;
        for (Member m : pop) {
            double f = 0;
            for (int k = 0; k < 3; k++) {
                f += fitness(m.program(), tasks.apply(population + cycles + k), seed + k);
            }
            track(m.program(), f / 3);
        }
        return best;
    }

    /** The largest scalar or vector address a program uses, so hand-written ones always fit. */
    private static int maxAddress(Program p, boolean scalar) {
        int max = 0;
        for (List<Instruction> code : List.of(p.setup(), p.predict(), p.learn())) {
            for (Instruction in : code) {
                boolean vOut = switch (in.op()) {
                    case V_ADD, V_SUB, V_MUL, SV_MUL, V_RANDOM -> true;
                    default -> false;
                };
                if (vOut != scalar) {
                    max = Math.max(max, in.out());
                }
                max = Math.max(max, Math.max(in.a(), in.b()));   // a loose bound is fine
            }
        }
        return max;
    }

    private void track(Program p, double f) {
        if (f > bestFitness) {
            bestFitness = f;
            best = p;
        }
    }

    private Program randomProgram(SplittableRandom r) {
        return new Program(randomCode(r, 1 + r.nextInt(3)), randomCode(r, 1 + r.nextInt(3)),
                randomCode(r, 1 + r.nextInt(3)));
    }

    private List<Instruction> randomCode(SplittableRandom r, int length) {
        List<Instruction> out = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            out.add(randomInstruction(r));
        }
        return out;
    }

    private Instruction randomInstruction(SplittableRandom r) {
        Op op = ops[r.nextInt(ops.length)];
        boolean vectorOut = switch (op) {
            case V_ADD, V_SUB, V_MUL, SV_MUL, V_RANDOM -> true;
            default -> false;
        };
        boolean scalarA = switch (op) {
            case S_ADD, S_SUB, S_MUL, S_DIV, S_ABS, S_TANH, SV_MUL -> true;
            default -> false;
        };
        boolean scalarB = switch (op) {
            case S_ADD, S_SUB, S_MUL, S_DIV -> true;
            default -> false;
        };
        int out = vectorOut ? 1 + r.nextInt(vectors - 1) : r.nextInt(scalars);
        int a = scalarA ? r.nextInt(scalars) : r.nextInt(vectors);
        int b = scalarB ? r.nextInt(scalars) : r.nextInt(vectors);
        double constant = Math.exp(r.nextDouble(Math.log(1e-3), Math.log(1)));
        if (op == Op.S_CONST && r.nextBoolean()) {
            constant = -constant;
        }
        return new Instruction(op, out, a, b, constant);
    }

    private Program mutate(Program p, SplittableRandom r) {
        List<List<Instruction>> parts = new ArrayList<>(List.of(new ArrayList<>(p.setup()),
                new ArrayList<>(p.predict()), new ArrayList<>(p.learn())));
        List<Instruction> code = parts.get(r.nextInt(3));
        switch (r.nextInt(4)) {
            case 0 -> {
                if (code.size() < MAX_LENGTH) {
                    code.add(r.nextInt(code.size() + 1), randomInstruction(r));
                }
            }
            case 1 -> {
                if (!code.isEmpty()) {
                    code.remove(r.nextInt(code.size()));
                }
            }
            case 2 -> {
                if (!code.isEmpty()) {
                    code.set(r.nextInt(code.size()), randomInstruction(r));
                }
            }
            default -> {
                code.clear();
                code.addAll(randomCode(r, 1 + r.nextInt(3)));
            }
        }
        return new Program(parts.get(0), parts.get(1), parts.get(2));
    }
}
