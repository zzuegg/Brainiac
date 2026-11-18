package io.github.zzuegg.brainiac.experimental.ops;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims.JobShopInstance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.LongFunction;

/**
 * Job-shop scheduling (minimise the makespan): every job runs its operations in a fixed machine
 * order, every machine runs one operation at a time.
 *
 * <ul>
 *   <li>{@link Builder}: Giffler-Thompson active-schedule generation; the choice set of every
 *       step is the conflict set on the machine of the earliest-completing operation, so every
 *       active schedule (an optimal one included) is reachable;</li>
 *   <li>{@link #dispatch}: priority dispatching rules SPT, MWKR, MOPNR, FDD/MWKR and random;</li>
 *   <li>{@link #branchAndBound}: depth-first branch and bound over the Giffler-Thompson tree with
 *       job and machine (head plus remaining load) bounds, exact on 6 x 6 within its node limit;</li>
 *   <li>{@link LearnedDispatcher}: L2D-style learned dispatching (Zhang et al., NeurIPS 2020,
 *       <a href="https://arxiv.org/abs/2010.12367">arXiv:2010.12367</a>): every operation of the
 *       disjunctive graph carries (completion-time lower bound, scheduled flag, processing time);
 *       a GIN-style layer sums each candidate's neighbourhood (its job predecessor and successor and
 *       the other unscheduled operations on its machine), adds a mean-pooled graph embedding, and a
 *       small MLP scores the candidates; the policy is trained with PPO-clip on random 6 x 6
 *       instances with a per-instance shared baseline, and generalises to larger instances because
 *       its features are size-free.</li>
 * </ul>
 */
public final class JobShop {

    private JobShop() {
    }

    /** Priority dispatching rules. */
    public enum Rule {
        /** shortest processing time */
        SPT,
        /** most work remaining */
        MWKR,
        /** most operations remaining */
        MOPNR,
        /** flow due date per most work remaining */
        FDD_MWKR,
        /** uniformly random */
        RANDOM
    }

    /** A partial schedule built by Giffler-Thompson steps. */
    public static final class Builder {
        private final JobShopInstance instance;
        private final int[] next;
        private final int[] jobReady;
        private final int[] machineReady;
        private final int[] remainingWork;
        private final int[] machineRemaining;
        private final int[][] end;
        private int done;

        /** @param instance the instance */
        public Builder(JobShopInstance instance) {
            this.instance = instance;
            int n = instance.jobs();
            int m = instance.machines();
            next = new int[n];
            jobReady = new int[n];
            machineReady = new int[m];
            remainingWork = new int[n];
            machineRemaining = new int[m];
            end = new int[n][m];
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < m; k++) {
                    remainingWork[j] += instance.time()[j][k];
                    machineRemaining[instance.machine()[j][k]] += instance.time()[j][k];
                    end[j][k] = -1;
                }
            }
        }

        private Builder(Builder b) {
            instance = b.instance;
            next = b.next.clone();
            jobReady = b.jobReady.clone();
            machineReady = b.machineReady.clone();
            remainingWork = b.remainingWork.clone();
            machineRemaining = b.machineRemaining.clone();
            end = new int[b.end.length][];
            for (int j = 0; j < end.length; j++) {
                end[j] = b.end[j].clone();
            }
            done = b.done;
        }

        /** @return a copy */
        public Builder copy() {
            return new Builder(this);
        }

        /** @return whether every operation is scheduled */
        public boolean finished() {
            return done == instance.jobs() * instance.machines();
        }

        private int est(int j) {
            return Math.max(jobReady[j], machineReady[instance.machine()[j][next[j]]]);
        }

        /** @return the jobs whose next operation is in the Giffler-Thompson conflict set */
        public int[] conflictSet() {
            int n = instance.jobs();
            int bestEct = Integer.MAX_VALUE;
            int bestMachine = -1;
            for (int j = 0; j < n; j++) {
                if (next[j] < instance.machines()) {
                    int ect = est(j) + instance.time()[j][next[j]];
                    if (ect < bestEct) {
                        bestEct = ect;
                        bestMachine = instance.machine()[j][next[j]];
                    }
                }
            }
            List<Integer> out = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (next[j] < instance.machines() && instance.machine()[j][next[j]] == bestMachine
                        && est(j) < bestEct) {
                    out.add(j);
                }
            }
            return out.stream().mapToInt(Integer::intValue).toArray();
        }

        /** @param job a job from the conflict set; its next operation is started as early as possible */
        public void schedule(int job) {
            int k = next[job];
            int mach = instance.machine()[job][k];
            int p = instance.time()[job][k];
            int start = est(job);
            end[job][k] = start + p;
            jobReady[job] = start + p;
            machineReady[mach] = start + p;
            remainingWork[job] -= p;
            machineRemaining[mach] -= p;
            next[job]++;
            done++;
        }

        /** @return the latest completion so far */
        public int makespan() {
            int ms = 0;
            for (int r : jobReady) {
                ms = Math.max(ms, r);
            }
            return ms;
        }

        /** @return a lower bound on the makespan of every completion of this partial schedule */
        public int lowerBound() {
            int n = instance.jobs();
            int m = instance.machines();
            int lb = makespan();
            int[] minHead = new int[m];
            Arrays.fill(minHead, Integer.MAX_VALUE);
            for (int j = 0; j < n; j++) {
                lb = Math.max(lb, jobReady[j] + remainingWork[j]);
                int head = jobReady[j];
                for (int k = next[j]; k < m; k++) {
                    int mach = instance.machine()[j][k];
                    minHead[mach] = Math.min(minHead[mach], head);
                    head += instance.time()[j][k];
                }
            }
            for (int mach = 0; mach < m; mach++) {
                if (machineRemaining[mach] > 0) {
                    lb = Math.max(lb, Math.max(machineReady[mach], minHead[mach]) + machineRemaining[mach]);
                }
            }
            return lb;
        }

        /**
         * @param job a job
         * @param rule a rule
         * @return the job's priority under the rule (higher first)
         */
        double priority(int job, Rule rule) {
            int k = next[job];
            int p = instance.time()[job][k];
            return switch (rule) {
                case SPT -> -p;
                case MWKR -> remainingWork[job];
                case MOPNR -> instance.machines() - k;
                case FDD_MWKR -> {
                    int doneWork = 0;
                    for (int i = 0; i <= k; i++) {
                        doneWork += instance.time()[job][i];
                    }
                    yield -(double) doneWork / Math.max(1, remainingWork[job]);
                }
                case RANDOM -> 0;
            };
        }
    }

    /**
     * Builds a schedule with a dispatching rule over the conflict sets.
     *
     * @param instance the instance
     * @param rule the rule
     * @param seed the seed (random rule and ties)
     * @return the makespan
     */
    public static int dispatch(JobShopInstance instance, Rule rule, long seed) {
        Objects.requireNonNull(rule, "rule");
        SplittableRandom r = new SplittableRandom(seed);
        Builder b = new Builder(instance);
        while (!b.finished()) {
            int[] c = b.conflictSet();
            int best = c[0];
            double bv = Double.NEGATIVE_INFINITY;
            for (int j : c) {
                double v = b.priority(j, rule) + r.nextDouble() * (rule == Rule.RANDOM ? 1 : 1e-9);
                if (v > bv) {
                    bv = v;
                    best = j;
                }
            }
            b.schedule(best);
        }
        return b.makespan();
    }

    /**
     * Depth-first branch and bound.
     *
     * @param instance the instance
     * @param nodeLimit the most nodes to expand
     * @return {best makespan, 1 if proved optimal else 0, nodes expanded}
     */
    public static long[] branchAndBound(JobShopInstance instance, long nodeLimit) {
        long best = Long.MAX_VALUE;
        for (Rule rule : Rule.values()) {
            best = Math.min(best, dispatch(instance, rule, 1));
        }
        long[] state = {best, 0};
        boolean complete = dfs(new Builder(instance), state, nodeLimit);
        return new long[] {state[0], complete ? 1 : 0, state[1]};
    }

    private static boolean dfs(Builder b, long[] state, long limit) {
        if (state[1]++ >= limit) {
            return false;
        }
        if (b.finished()) {
            state[0] = Math.min(state[0], b.makespan());
            return true;
        }
        if (b.lowerBound() >= state[0]) {
            return true;
        }
        int[] c = b.conflictSet();
        Integer[] order = new Integer[c.length];
        for (int i = 0; i < c.length; i++) {
            order[i] = c[i];
        }
        Arrays.sort(order, (x, y) -> Double.compare(b.priority(y, Rule.MWKR), b.priority(x, Rule.MWKR)));
        boolean complete = true;
        for (int j : order) {
            Builder child = b.copy();
            child.schedule(j);
            if (!dfs(child, state, limit)) {
                complete = false;
                break;
            }
        }
        return complete;
    }

    /** The L2D-style learned dispatcher. */
    public static final class LearnedDispatcher {
        private static final int RAW = 3;
        private static final int IN = 3 * RAW;
        private final int hidden;
        private final double[] params;
        private final double[] m1;
        private final double[] m2;
        private long adamStep;
        private final SplittableRandom random;
        private double clip = 0.2;
        private double learningRate = 3e-3;

        private LearnedDispatcher(int hidden, long seed) {
            this.hidden = hidden;
            this.random = new SplittableRandom(seed);
            this.params = new double[hidden * IN + hidden + hidden + 1];
            double s = 1 / Math.sqrt(IN);
            for (int i = 0; i < hidden * IN; i++) {
                params[i] = random.nextGaussian() * s;
            }
            for (int h = 0; h < hidden; h++) {
                params[hidden * IN + hidden + h] = random.nextGaussian() * 0.1;
            }
            m1 = new double[params.length];
            m2 = new double[params.length];
        }

        /**
         * @param hidden hidden units of the scoring MLP
         * @param seed the seed
         * @return an untrained dispatcher
         */
        public static LearnedDispatcher of(int hidden, long seed) {
            return new LearnedDispatcher(hidden, seed);
        }

        /** Candidate inputs: the operation's raw features, its neighbourhood mean, the graph mean. */
        private double[][] features(Builder b, int[] candidates) {
            JobShopInstance inst = b.instance;
            int n = inst.jobs();
            int m = inst.machines();
            int maxT = 1;
            double total = 0;
            for (int[] row : inst.time()) {
                for (int t : row) {
                    maxT = Math.max(maxT, t);
                    total += t;
                }
            }
            double scale = total / m * 2;
            double[][][] raw = new double[n][m][RAW];
            double[] mean = new double[RAW];
            for (int j = 0; j < n; j++) {
                double lb = 0;
                for (int k = 0; k < m; k++) {
                    int p = inst.time()[j][k];
                    if (b.end[j][k] >= 0) {
                        lb = b.end[j][k];
                    } else if (k == b.next[j]) {
                        lb = b.est(j) + p;
                    } else {
                        lb += p;
                    }
                    raw[j][k][0] = lb / scale;
                    raw[j][k][1] = b.end[j][k] >= 0 ? 1 : 0;
                    raw[j][k][2] = p / (double) maxT;
                    for (int f = 0; f < RAW; f++) {
                        mean[f] += raw[j][k][f] / (n * m);
                    }
                }
            }
            double[][] out = new double[candidates.length][IN];
            for (int c = 0; c < candidates.length; c++) {
                int j = candidates[c];
                int k = b.next[j];
                int mach = inst.machine()[j][k];
                double[] agg = new double[RAW];
                int count = 0;
                if (k > 0) {
                    add(agg, raw[j][k - 1]);
                    count++;
                }
                if (k + 1 < m) {
                    add(agg, raw[j][k + 1]);
                    count++;
                }
                for (int jj = 0; jj < n; jj++) {
                    for (int kk = b.next[jj]; kk < m; kk++) {
                        if (jj != j && inst.machine()[jj][kk] == mach) {
                            add(agg, raw[jj][kk]);
                            count++;
                        }
                    }
                }
                for (int f = 0; f < RAW; f++) {
                    out[c][f] = raw[j][k][f];
                    out[c][RAW + f] = count == 0 ? 0 : agg[f] / count;
                    out[c][2 * RAW + f] = mean[f];
                }
            }
            return out;
        }

        private static void add(double[] a, double[] b) {
            for (int i = 0; i < a.length; i++) {
                a[i] += b[i];
            }
        }

        private double score(double[] z, double[] hiddenOut) {
            double s = params[params.length - 1];
            for (int h = 0; h < hidden; h++) {
                double a = params[hidden * IN + h];
                for (int i = 0; i < IN; i++) {
                    a += params[h * IN + i] * z[i];
                }
                double t = Math.tanh(a);
                if (hiddenOut != null) {
                    hiddenOut[h] = t;
                }
                s += params[hidden * IN + hidden + h] * t;
            }
            return s;
        }

        private double[] probabilities(double[][] z) {
            double[] s = new double[z.length];
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < z.length; c++) {
                s[c] = score(z[c], null);
                max = Math.max(max, s[c]);
            }
            double sum = 0;
            for (int c = 0; c < s.length; c++) {
                s[c] = Math.exp(s[c] - max);
                sum += s[c];
            }
            for (int c = 0; c < s.length; c++) {
                s[c] /= sum;
            }
            return s;
        }

        private record Decision(double[][] z, int choice, double oldLogProb) {
        }

        private int rollout(JobShopInstance inst, boolean sample, List<Decision> record) {
            Builder b = new Builder(inst);
            while (!b.finished()) {
                int[] c = b.conflictSet();
                int pick = 0;
                if (c.length > 1) {
                    double[][] z = features(b, c);
                    double[] p = probabilities(z);
                    if (sample) {
                        double u = random.nextDouble();
                        pick = c.length - 1;
                        for (int i = 0; i < c.length; i++) {
                            u -= p[i];
                            if (u <= 0) {
                                pick = i;
                                break;
                            }
                        }
                    } else {
                        for (int i = 1; i < c.length; i++) {
                            if (p[i] > p[pick]) {
                                pick = i;
                            }
                        }
                    }
                    if (record != null) {
                        record.add(new Decision(z, pick, Math.log(p[pick])));
                    }
                }
                b.schedule(c[pick]);
            }
            return b.makespan();
        }

        /**
         * Greedy (argmax) dispatching.
         *
         * @param instance the instance
         * @return the makespan
         */
        public int solve(JobShopInstance instance) {
            return rollout(instance, false, null);
        }

        /**
         * Trains with PPO-clip.
         *
         * @param instances a generator of training instances by seed
         * @param iterations PPO iterations
         * @param instancesPerIteration instances per iteration
         * @param rollouts sampled schedules per instance (they share a baseline)
         * @param epochs gradient epochs per iteration
         * @return this dispatcher
         */
        public LearnedDispatcher train(LongFunction<JobShopInstance> instances, int iterations,
                int instancesPerIteration, int rollouts, int epochs) {
            long seed = 0;
            for (int it = 0; it < iterations; it++) {
                List<Decision> decisions = new ArrayList<>();
                List<Double> advantages = new ArrayList<>();
                for (int i = 0; i < instancesPerIteration; i++) {
                    JobShopInstance inst = instances.apply(seed++);
                    double[] ret = new double[rollouts];
                    List<List<Decision>> recs = new ArrayList<>();
                    for (int k = 0; k < rollouts; k++) {
                        List<Decision> rec = new ArrayList<>();
                        ret[k] = -rollout(inst, true, rec);
                        recs.add(rec);
                    }
                    double mean = Arrays.stream(ret).average().orElse(0);
                    double sd = Math.sqrt(Arrays.stream(ret).map(x -> (x - mean) * (x - mean)).average().orElse(0)) + 1e-6;
                    for (int k = 0; k < rollouts; k++) {
                        for (Decision d : recs.get(k)) {
                            decisions.add(d);
                            advantages.add((ret[k] - mean) / sd);
                        }
                    }
                }
                for (int e = 0; e < epochs; e++) {
                    double[] grad = new double[params.length];
                    for (int idx = 0; idx < decisions.size(); idx++) {
                        accumulate(decisions.get(idx), advantages.get(idx), grad);
                    }
                    adam(grad, decisions.size());
                }
            }
            return this;
        }

        private void accumulate(Decision d, double adv, double[] grad) {
            double[] p = probabilities(d.z());
            double ratio = Math.exp(Math.log(p[d.choice()]) - d.oldLogProb());
            if ((adv > 0 && ratio > 1 + clip) || (adv < 0 && ratio < 1 - clip)) {
                return;
            }
            double[] hid = new double[hidden];
            for (int c = 0; c < d.z().length; c++) {
                double coeff = ratio * adv * ((c == d.choice() ? 1 : 0) - p[c]);
                if (coeff == 0) {
                    continue;
                }
                double[] z = d.z()[c];
                score(z, hid);
                int w2 = hidden * IN + hidden;
                grad[params.length - 1] += coeff;
                for (int h = 0; h < hidden; h++) {
                    grad[w2 + h] += coeff * hid[h];
                    double back = coeff * params[w2 + h] * (1 - hid[h] * hid[h]);
                    grad[hidden * IN + h] += back;
                    for (int i = 0; i < IN; i++) {
                        grad[h * IN + i] += back * z[i];
                    }
                }
            }
        }

        private void adam(double[] grad, int n) {
            adamStep++;
            for (int i = 0; i < params.length; i++) {
                double g = grad[i] / Math.max(1, n);
                m1[i] = 0.9 * m1[i] + 0.1 * g;
                m2[i] = 0.999 * m2[i] + 0.001 * g * g;
                double mh = m1[i] / (1 - Math.pow(0.9, adamStep));
                double vh = m2[i] / (1 - Math.pow(0.999, adamStep));
                params[i] += learningRate * mh / (Math.sqrt(vh) + 1e-8);
            }
        }
    }
}
