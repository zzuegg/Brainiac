package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * POMO: policy optimisation with multiple optima, on an {@link AttentionModel}. Every
 * instance is rolled out {@code N} times, each rollout forced to start from a different
 * node (TSP) or first customer (CVRP); the mean length of the {@code N} rollouts is the
 * shared REINFORCE baseline, so no critic and no baseline policy are needed. At inference
 * the best of the {@code N} greedy rollouts is taken, optionally also over the eight
 * coordinate symmetries of the unit square ({@link RoutingEnv#symmetry}).
 *
 * <p>The Sym-NCO option ({@link #symNco}) rolls each instance out on {@code K} random
 * rotations as well, with one baseline shared over rotations and starts (problem symmetry),
 * and adds a loss {@code -cos(g(h(x)), g(h(Qx)))} that makes a projection of the mean node
 * embedding invariant to the rotation {@code Q}.
 *
 * <p>Papers: Kwon, Choo, Kim, Yoon, Gwon and Min, "POMO: Policy Optimization with Multiple
 * Optima for Reinforcement Learning" (NeurIPS 2020),
 * <a href="https://arxiv.org/abs/2010.16011">arXiv:2010.16011</a>; Kim, Park and Park,
 * "Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization" (NeurIPS 2022),
 * <a href="https://arxiv.org/abs/2205.13209">arXiv:2205.13209</a>.
 */
public final class Pomo {

    /**
     * Multi-start rollouts of a batch of instances.
     *
     * @param rollout the {@code instances * starts} rollouts, grouped by instance
     * @param envs the environments rolled out (replicated per start)
     * @param starts the forced first nodes
     * @param group the instance group of each rollout (rotations of one instance share one)
     * @param scale each rollout's length scale relative to its original instance
     */
    public record MultiStart(AttentionModel.Rollout rollout, List<RoutingEnv> envs, int[] starts, int[] group,
            double[] scale) {
        /**
         * @return each rollout's length divided by its scale (comparable within a group)
         */
        public double[] normalizedLengths() {
            double[] l = rollout.lengths().clone();
            for (int i = 0; i < l.length; i++) {
                l[i] /= scale[i];
            }
            return l;
        }

        /** @return the shared baseline: the mean normalized length of each rollout's group */
        public double[] sharedBaseline() {
            double[] l = normalizedLengths();
            int groups = 0;
            for (int g : group) {
                groups = Math.max(groups, g + 1);
            }
            double[] sum = new double[groups];
            int[] count = new int[groups];
            for (int i = 0; i < l.length; i++) {
                sum[group[i]] += l[i];
                count[group[i]]++;
            }
            double[] b = new double[l.length];
            for (int i = 0; i < l.length; i++) {
                b[i] = sum[group[i]] / count[group[i]] * scale[i];
            }
            return b;
        }
    }

    private final AttentionModel model;
    private double symWeight;
    private int rotations;
    private int maxStarts = Integer.MAX_VALUE;
    private final Var projection;

    private Pomo(AttentionModel model) {
        this.model = model;
        this.projection = AttentionModel.matrix(new SplittableRandom(17), model.dim(), model.dim());
    }

    /**
     * POMO on a model.
     *
     * @param model the policy
     * @return the trainer and solver
     */
    public static Pomo of(AttentionModel model) {
        return new Pomo(model);
    }

    /**
     * Turns on Sym-NCO.
     *
     * @param weight the invariance loss weight (0 keeps only the shared problem-symmetry
     *     baseline)
     * @param rotations random rotations per instance in training
     * @return this
     */
    public Pomo symNco(double weight, int rotations) {
        this.symWeight = weight;
        this.rotations = rotations;
        return this;
    }

    /**
     * @param n the most starts per instance (fewer than the nodes subsamples them evenly)
     * @return this
     */
    public Pomo maxStarts(int n) {
        this.maxStarts = n;
        return this;
    }

    /** @return the policy */
    public AttentionModel model() {
        return model;
    }

    /**
     * The start nodes of an instance: every node (TSP) or every customer (CVRP), evenly
     * subsampled to {@link #maxStarts}.
     *
     * @param env the instance
     * @return the first nodes
     */
    public int[] startNodes(RoutingEnv env) {
        int lo = env.kind() == RoutingEnv.Kind.TSP ? 0 : 1;
        int all = env.nodes() - lo;
        int k = Math.min(all, maxStarts);
        int[] s = new int[k];
        for (int i = 0; i < k; i++) {
            s[i] = lo + (int) ((long) i * all / k);
        }
        return s;
    }

    /**
     * Samples multi-start rollouts (with gradients).
     *
     * @param instances the instances
     * @param decode sampled (training) or greedy
     * @param random the source
     * @return the rollouts
     */
    public MultiStart rollouts(List<RoutingEnv> instances, AttentionModel.Decode decode, SplittableRandom random) {
        List<RoutingEnv> envs = new ArrayList<>();
        List<Integer> starts = new ArrayList<>();
        List<Integer> groups = new ArrayList<>();
        List<Double> scales = new ArrayList<>();
        for (int g = 0; g < instances.size(); g++) {
            RoutingEnv base = instances.get(g);
            List<RoutingEnv> versions = new ArrayList<>(List.of(base));
            for (int k = 0; k < rotations; k++) {
                versions.add(base.rotated(2 * Math.PI * random.nextDouble()));
            }
            for (RoutingEnv v : versions) {
                for (int s : startNodes(v)) {
                    envs.add(v);
                    starts.add(s);
                    groups.add(g);
                    scales.add(v.scaleFrom(base));
                }
            }
        }
        int[] st = starts.stream().mapToInt(Integer::intValue).toArray();
        AttentionModel.Rollout ro = model.rollout(envs, decode, st, random);
        return new MultiStart(ro, envs, st, groups.stream().mapToInt(Integer::intValue).toArray(),
                scales.stream().mapToDouble(Double::doubleValue).toArray());
    }

    /**
     * One POMO training step.
     *
     * @param instances the instances
     * @param optimizer the optimizer
     * @param random the source
     * @return the number of rollouts used
     */
    public int trainStep(List<RoutingEnv> instances, Optimizer optimizer, SplittableRandom random) {
        MultiStart ms = rollouts(instances, AttentionModel.Decode.SAMPLE, random);
        int b = ms.scale().length;
        Var loss = ms.rollout().logp().mul(Tensor.of(Shape.of(b, 1), advantages(ms))).mean();
        List<Var> params = new ArrayList<>(model.parameters());
        if (symWeight > 0 && rotations > 0) {
            Var z = ms.rollout().graph().matmul(projection).normalize(1, 1e-6f);   // [b, d]
            int per = startNodes(instances.get(0)).length;
            int versions = rotations + 1;
            Var inv2 = null;
            for (int g = 0; g < instances.size(); g++) {
                int base = g * versions * per;
                Var anchor = z.slice(0, base, base + 1);
                for (int k = 1; k < versions; k++) {
                    int row = base + k * per;
                    Var cos = anchor.mul(z.slice(0, row, row + 1)).sum();
                    inv2 = inv2 == null ? cos : inv2.add(cos);
                }
            }
            loss = loss.sub(inv2.mul((float) (symWeight / (instances.size() * rotations))));
            params.add(projection);
        }
        AttentionModel.step(loss, params, optimizer, 1.0);
        return b;
    }

    /**
     * The scale-normalized advantages {@code L / s - mean_group(L / s)}.
     *
     * @param ms the rollouts
     * @return one advantage per rollout
     */
    static float[] advantages(MultiStart ms) {
        double[] l = ms.normalizedLengths();
        double[] base = ms.sharedBaseline();
        float[] adv = new float[l.length];
        for (int i = 0; i < l.length; i++) {
            adv[i] = (float) (l[i] - base[i] / ms.scale()[i]);
        }
        return adv;
    }

    /**
     * Solves one instance: the best greedy rollout over all starts, and over the eight
     * symmetries when {@code augment} is set.
     *
     * @param env the instance
     * @param augment use the x8 augmentation
     * @return the best solution
     */
    public int[] solve(RoutingEnv env, boolean augment) {
        List<RoutingEnv> envs = new ArrayList<>();
        List<Integer> starts = new ArrayList<>();
        for (int k = 0; k < (augment ? 8 : 1); k++) {
            RoutingEnv v = env.symmetry(k);
            for (int s : startNodes(v)) {
                envs.add(v);
                starts.add(s);
            }
        }
        AttentionModel.Rollout ro = Var.noGrad(() -> model.rollout(envs, AttentionModel.Decode.GREEDY,
                starts.stream().mapToInt(Integer::intValue).toArray(), null));
        int best = 0;
        for (int i = 1; i < ro.lengths().length; i++) {
            if (ro.lengths()[i] < ro.lengths()[best]) {
                best = i;
            }
        }
        return ro.solutions()[best];
    }
}
