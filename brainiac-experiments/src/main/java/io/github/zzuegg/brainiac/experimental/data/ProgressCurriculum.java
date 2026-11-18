package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learning-progress curricula: a teacher picks the next task by how fast the learner is
 * improving on it, not by a fixed difficulty order.
 * <ul>
 *   <li>{@link #exp3s}: Graves et al. ("Automated Curriculum Learning for Neural Networks",
 *       ICML 2017, <a href="https://arxiv.org/abs/1704.03003">arXiv:1704.03003</a>): a
 *       non-stationary EXP3.S bandit whose reward is the progress on the chosen task (its score
 *       gain since it was last trained), rescaled to [-1, 1] by the 20th and 80th percentiles of
 *       recent rewards;</li>
 *   <li>{@link #slopeGreedy} and {@link #slopeThompson}: Matiisen et al. ("Teacher-Student
 *       Curriculum Learning", 2017, <a href="https://arxiv.org/abs/1707.00183">arXiv:1707.00183</a>):
 *       the absolute slope of each task's recent learning curve, so tasks being learnt or being
 *       forgotten get attention and mastered or impossible ones do not; picked ε-greedily or by
 *       Thompson-style sampling from each task's recent score changes.</li>
 * </ul>
 * Use: {@link #next()} gives the task to train on, then {@link #observe} reports the learner's
 * score on that task (higher is better, for example accuracy or negative loss).
 */
public final class ProgressCurriculum {

    /** The teacher. */
    public enum Teacher {
        /** EXP3.S on progress. */
        EXP3S,
        /** ε-greedy on the absolute learning-curve slope. */
        SLOPE_GREEDY,
        /** Thompson-style sampling of recent absolute score changes. */
        SLOPE_THOMPSON
    }

    private final Teacher teacher;
    private final int tasks;
    private final double eta;
    private final double epsilon;
    private final double alpha;
    private final int window;
    private final SplittableRandom random;
    private final double[] logWeights;
    private final double[] lastScore;
    private final List<Deque<double[]>> curves = new ArrayList<>();
    private final Deque<Double> rewards = new ArrayDeque<>();
    private final int[] counts;
    private double[] probabilities;
    private long step;

    private ProgressCurriculum(Teacher teacher, int tasks, double eta, double epsilon, double alpha, int window,
            long seed) {
        this.teacher = teacher;
        this.tasks = tasks;
        this.eta = eta;
        this.epsilon = epsilon;
        this.alpha = alpha;
        this.window = window;
        this.random = new SplittableRandom(seed);
        this.logWeights = new double[tasks];
        this.lastScore = new double[tasks];
        Arrays.fill(lastScore, Double.NaN);
        this.counts = new int[tasks];
        for (int i = 0; i < tasks; i++) {
            curves.add(new ArrayDeque<>());
        }
        this.probabilities = new double[tasks];
        Arrays.fill(probabilities, 1.0 / tasks);
    }

    /**
     * The EXP3.S teacher.
     *
     * @param tasks the number of tasks
     * @param eta the step size, for example 0.1
     * @param epsilon the exploration share, for example 0.05
     * @param alpha the weight sharing of EXP3.S, for example 0.01
     * @param seed the seed
     * @return the curriculum
     */
    public static ProgressCurriculum exp3s(int tasks, double eta, double epsilon, double alpha, long seed) {
        return new ProgressCurriculum(Teacher.EXP3S, tasks, eta, epsilon, alpha, 0, seed);
    }

    /**
     * The ε-greedy slope teacher.
     *
     * @param tasks the number of tasks
     * @param window recent observations per task for the slope, for example 5
     * @param epsilon the exploration share, for example 0.1
     * @param seed the seed
     * @return the curriculum
     */
    public static ProgressCurriculum slopeGreedy(int tasks, int window, double epsilon, long seed) {
        return new ProgressCurriculum(Teacher.SLOPE_GREEDY, tasks, 0, epsilon, 0, window, seed);
    }

    /**
     * The Thompson-style slope teacher.
     *
     * @param tasks the number of tasks
     * @param window recent score changes kept per task, for example 5
     * @param seed the seed
     * @return the curriculum
     */
    public static ProgressCurriculum slopeThompson(int tasks, int window, long seed) {
        return new ProgressCurriculum(Teacher.SLOPE_THOMPSON, tasks, 0, 0, 0, window, seed);
    }

    /** @return the teacher */
    public Teacher teacher() {
        return teacher;
    }

    /** @return how often each task was picked */
    public int[] counts() {
        return counts.clone();
    }

    /** @return the current sampling probabilities (EXP3.S) or the uniform vector */
    public double[] probabilities() {
        return probabilities.clone();
    }

    /** @return the task to train on next */
    public int next() {
        step++;
        int pick = switch (teacher) {
            case EXP3S -> {
                double max = Arrays.stream(logWeights).max().orElse(0);
                double sum = 0;
                double[] w = new double[tasks];
                for (int i = 0; i < tasks; i++) {
                    w[i] = Math.exp(logWeights[i] - max);
                    sum += w[i];
                }
                for (int i = 0; i < tasks; i++) {
                    probabilities[i] = (1 - epsilon) * w[i] / sum + epsilon / tasks;
                }
                yield draw(probabilities);
            }
            case SLOPE_GREEDY -> {
                int unseen = firstUnderObserved();
                if (unseen >= 0) {
                    yield unseen;
                }
                if (random.nextDouble() < epsilon) {
                    yield random.nextInt(tasks);
                }
                int best = 0;
                for (int i = 1; i < tasks; i++) {
                    best = Math.abs(slope(i)) > Math.abs(slope(best)) ? i : best;
                }
                yield best;
            }
            case SLOPE_THOMPSON -> {
                int unseen = firstUnderObserved();
                if (unseen >= 0) {
                    yield unseen;
                }
                int best = 0;
                double bestDraw = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < tasks; i++) {
                    Deque<double[]> c = curves.get(i);
                    double[][] pts = c.toArray(new double[0][]);
                    int k = 1 + random.nextInt(pts.length - 1);
                    double d = Math.abs(pts[k][1] - pts[k - 1][1]);
                    if (d > bestDraw) {
                        bestDraw = d;
                        best = i;
                    }
                }
                yield best;
            }
        };
        counts[pick]++;
        return pick;
    }

    private int firstUnderObserved() {
        for (int i = 0; i < tasks; i++) {
            if (curves.get(i).size() < 2) {
                return i;
            }
        }
        return -1;
    }

    private int draw(double[] p) {
        double u = random.nextDouble();
        for (int i = 0; i < p.length - 1; i++) {
            u -= p[i];
            if (u < 0) {
                return i;
            }
        }
        return p.length - 1;
    }

    private double slope(int task) {
        Deque<double[]> c = curves.get(task);
        int n = c.size();
        if (n < 2) {
            return Double.POSITIVE_INFINITY;
        }
        double mx = 0;
        double my = 0;
        for (double[] p : c) {
            mx += p[0] / n;
            my += p[1] / n;
        }
        double sxy = 0;
        double sxx = 0;
        for (double[] p : c) {
            sxy += (p[0] - mx) * (p[1] - my);
            sxx += (p[0] - mx) * (p[0] - mx);
        }
        return sxx == 0 ? 0 : sxy / sxx;
    }

    /**
     * Reports the learner's score on a task after training on it.
     *
     * @param task the task just trained
     * @param score its score now (higher is better)
     */
    public void observe(int task, double score) {
        double progress = Double.isNaN(lastScore[task]) ? 0 : score - lastScore[task];
        lastScore[task] = score;
        Deque<double[]> c = curves.get(task);
        c.addLast(new double[] {step, score});
        while (window > 0 && c.size() > window) {
            c.removeFirst();
        }
        if (teacher == Teacher.EXP3S) {
            rewards.addLast(progress);
            while (rewards.size() > 200) {
                rewards.removeFirst();
            }
            double[] sorted = rewards.stream().mapToDouble(Double::doubleValue).sorted().toArray();
            double lo = sorted[(int) (0.2 * (sorted.length - 1))];
            double hi = sorted[(int) (0.8 * (sorted.length - 1))];
            double r = hi == lo ? 0 : Math.max(-1, Math.min(1, 2 * (progress - lo) / (hi - lo) - 1));
            double sum = 0;
            double max = Arrays.stream(logWeights).max().orElse(0);
            for (double w : logWeights) {
                sum += Math.exp(w - max);
            }
            logWeights[task] += eta * r / Math.max(1e-6, probabilities[task]);
            double share = alpha * sum / tasks;
            for (int i = 0; i < tasks; i++) {
                double w = Math.exp(logWeights[i] - max);
                logWeights[i] = Math.log((1 - alpha) * w + share) + max;
            }
        }
    }

    @Override
    public String toString() {
        return "ProgressCurriculum(" + teacher + ", counts " + Arrays.toString(counts) + ")";
    }
}
