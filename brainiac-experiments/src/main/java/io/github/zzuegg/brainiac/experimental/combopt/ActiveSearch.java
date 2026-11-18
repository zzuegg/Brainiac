package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Test-time search for a trained constructive policy: adapt the {@link AttentionModel} to the
 * one instance in front of it. Each iteration samples POMO's multi-start rollouts, keeps the
 * best solution seen, and takes a gradient step on REINFORCE with the shared baseline plus
 * (for EAS) an imitation term {@code -lambda log pi(best so far)}:
 * <ul>
 *   <li>{@link Mode#SAMPLING}: the same rollouts with no updates (the baseline);</li>
 *   <li>{@link Mode#FULL}: active search (Bello et al.): fine-tune all weights;</li>
 *   <li>{@link Mode#EAS_EMB}: update only the node embeddings (the encoder runs once);</li>
 *   <li>{@link Mode#EAS_LAY}: insert a residual layer {@code q + relu(q W1 + b1) W2} with
 *       {@code W2 = 0} into the decoder and train only it;</li>
 *   <li>{@link Mode#EAS_TAB}: learn a table of logit offsets per (current, next) node pair.</li>
 * </ul>
 *
 * <p>Papers: Bello et al., "Neural Combinatorial Optimization with Reinforcement Learning"
 * (2017), <a href="https://arxiv.org/abs/1611.09940">arXiv:1611.09940</a>; Hottung, Kwon and
 * Tierney, "Efficient Active Search for Combinatorial Optimization Problems" (ICLR 2022),
 * <a href="https://arxiv.org/abs/2106.05126">arXiv:2106.05126</a>.
 */
public final class ActiveSearch {

    /** What is adapted. */
    public enum Mode {
        /** plain sampling, no adaptation */
        SAMPLING,
        /** all weights */
        FULL,
        /** the node embeddings */
        EAS_EMB,
        /** an added residual decoder layer */
        EAS_LAY,
        /** a table of logit offsets */
        EAS_TAB
    }

    /**
     * A search's outcome.
     *
     * @param best the best solution
     * @param length its length
     * @param rollouts rollouts sampled
     * @param curve the best length after each iteration
     */
    public record Result(int[] best, double length, int rollouts, double[] curve) {
    }

    private final AttentionModel trained;
    private final Mode mode;
    private float learningRate = 3e-3f;
    private float imitation = 0.05f;
    private int maxStarts = Integer.MAX_VALUE;

    private ActiveSearch(AttentionModel trained, Mode mode) {
        this.trained = trained;
        this.mode = mode;
    }

    /**
     * A searcher; the trained model is never changed (FULL works on a copy).
     *
     * @param trained the trained policy
     * @param mode what to adapt
     * @return the searcher
     */
    public static ActiveSearch of(AttentionModel trained, Mode mode) {
        return new ActiveSearch(trained, mode);
    }

    /**
     * @param lr the Adam rate of the adapted values
     * @return this
     */
    public ActiveSearch learningRate(float lr) {
        this.learningRate = lr;
        return this;
    }

    /**
     * @param lambda the imitation weight (EAS modes)
     * @return this
     */
    public ActiveSearch imitation(float lambda) {
        this.imitation = lambda;
        return this;
    }

    /**
     * @param n the most POMO starts per iteration
     * @return this
     */
    public ActiveSearch maxStarts(int n) {
        this.maxStarts = n;
        return this;
    }

    /**
     * Searches one instance.
     *
     * @param env the instance
     * @param iterations iterations of one multi-start batch each
     * @param random the source
     * @return the outcome
     */
    public Result search(RoutingEnv env, int iterations, SplittableRandom random) {
        AttentionModel model = mode == Mode.FULL ? trained.copy() : trained;
        Pomo pomo = Pomo.of(model).maxStarts(maxStarts);
        int[] starts = pomo.startNodes(env);
        int n = starts.length;
        List<RoutingEnv> envs = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            envs.add(env);
        }
        int nodes = env.nodes();
        int dim = model.dim();
        SplittableRandom init = new SplittableRandom(3);
        Var base = mode == Mode.FULL ? null : Var.noGrad(() -> model.encode(List.of(env))).detach();
        Var delta = Var.param(Tensor.zeros(Shape.of(1, nodes, dim)));
        Var w1 = AttentionModel.matrix(init, dim, dim);
        Var b1 = Var.param(Tensor.zeros(Shape.of(dim)));
        Var w2 = Var.param(Tensor.zeros(Shape.of(dim, dim)));
        Var table = Var.param(Tensor.zeros(Shape.of(nodes, nodes)));
        List<Var> params = switch (mode) {
            case SAMPLING -> List.of();
            case FULL -> model.parameters();
            case EAS_EMB -> List.of(delta);
            case EAS_LAY -> List.of(w1, b1, w2);
            case EAS_TAB -> List.of(table);
        };
        AttentionModel.Adapter adapter = switch (mode) {
            case EAS_LAY -> new AttentionModel.Adapter() {
                @Override
                public Var query(Var q) {
                    return q.add(q.matmul(w1).add(b1).relu().matmul(w2));
                }
            };
            case EAS_TAB -> new AttentionModel.Adapter() {
                @Override
                public Var logitBias(int[] current) {
                    float[] idx = new float[current.length];
                    for (int i = 0; i < idx.length; i++) {
                        idx[i] = Math.max(0, current[i]);
                    }
                    return table.gather(0, Tensor.of(Shape.of(idx.length), idx));
                }
            };
            default -> null;
        };
        Optimizer opt = Optimizer.adam(learningRate);
        int[] best = null;
        double bestLen = Double.POSITIVE_INFINITY;
        double[] curve = new double[iterations];
        for (int it = 0; it < iterations; it++) {
            Var emb = base == null ? null
                    : (mode == Mode.EAS_EMB ? base.add(delta) : base).expand(Shape.of(n, nodes, dim));
            boolean grad = mode != Mode.SAMPLING;
            AttentionModel.Rollout ro = grad
                    ? model.rollout(envs, emb, AttentionModel.Decode.SAMPLE, starts, null, adapter, random)
                    : Var.noGrad(() -> model.rollout(envs, emb, AttentionModel.Decode.SAMPLE, starts, null, adapter, random));
            double mean = Arrays.stream(ro.lengths()).average().orElse(0);
            for (int i = 0; i < n; i++) {
                if (ro.lengths()[i] < bestLen) {
                    bestLen = ro.lengths()[i];
                    best = ro.solutions()[i];
                }
            }
            curve[it] = bestLen;
            if (!grad) {
                continue;
            }
            float[] adv = new float[n];
            for (int i = 0; i < n; i++) {
                adv[i] = (float) (ro.lengths()[i] - mean);
            }
            Var loss = ro.logp().mul(Tensor.of(Shape.of(n, 1), adv)).mean();
            if (mode != Mode.FULL && imitation > 0) {
                int[] tour = best;
                int first = env.kind() == RoutingEnv.Kind.TSP ? tour[0] : firstCustomer(tour);
                Var one = base == null ? null : (mode == Mode.EAS_EMB ? base.add(delta) : base);
                AttentionModel.Rollout il = model.rollout(List.of(env), one, AttentionModel.Decode.GREEDY,
                        new int[] {first}, new int[][] {tour}, adapter, random);
                loss = loss.sub(il.logp().mean().mul(imitation));
            }
            AttentionModel.step(loss, params, opt, 1.0);
            if (mode != Mode.FULL) {
                model.parameters().forEach(Var::zeroGrad);
            }
        }
        return new Result(best, bestLen, iterations * n, curve);
    }

    private static int firstCustomer(int[] route) {
        for (int v : route) {
            if (v != 0) {
                return v;
            }
        }
        return 1;
    }
}
