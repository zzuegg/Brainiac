package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.EntropicTransport;
import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * An unsupervised non-autoregressive TSP solver in the style of UTSP and "Graph neural
 * networks are heuristics": a network assigns every node a soft position on the tour, with no
 * labels and no rewards. A {@link TransformerBlock} encoder maps each node to {@code 2H}
 * numbers read as Fourier coefficients of a periodic score over the tour positions,
 * {@code s_ik = sum_h a_ih cos(2 pi h k / n) + b_ih sin(2 pi h k / n)}, so one network serves
 * every size {@code n}. Gumbel noise and log-domain Sinkhorn iterations turn the scores into a
 * doubly stochastic soft permutation {@code P} (node {@code i} at position {@code k}), and the
 * loss is the expected tour length {@code sum_ijk P_ik P_j(k+1) D'_ij} with
 * {@code D' = D + lambda I} (the diagonal penalty stops a node from filling two consecutive
 * positions for free).
 *
 * <p>Decoding sharpens the scores into a coupling with {@link EntropicTransport#sinkhorn}
 * and takes the assignment with the Hungarian solver of
 * {@link MinibatchOptimalTransport#assignCost}; the tour visits nodes in position order.
 * Further candidates come from MC dropout in the encoder plus Gumbel noise on the coupling,
 * and the shortest is kept.
 *
 * <p>Papers: Min, Bai and Gomes, "Unsupervised Learning for Solving the Travelling Salesman
 * Problem" (NeurIPS 2023), <a href="https://arxiv.org/abs/2303.10538">arXiv:2303.10538</a>; Min
 * and Gomes, "Graph Neural Networks are Heuristics" (2026),
 * <a href="https://arxiv.org/abs/2601.13465">arXiv:2601.13465</a>; Mena et al., "Learning Latent
 * Permutations with Gumbel-Sinkhorn Networks" (ICLR 2018),
 * <a href="https://arxiv.org/abs/1802.08665">arXiv:1802.08665</a>.
 */
public final class SoftPermutationTsp {

    private final int harmonics;
    private final Var embed;
    private final Var embedBias;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final Var head;
    private float temperature = 0.3f;
    private int sinkhornIterations = 20;

    private SoftPermutationTsp(int dim, int layers, int harmonics, float dropout, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.harmonics = harmonics;
        this.embed = AttentionModel.matrix(r, 2, dim);
        this.embedBias = Var.param(Tensor.zeros(Shape.of(dim)));
        for (int l = 0; l < layers; l++) {
            blocks.add((TransformerBlock) TransformerBlock.of(4).ffnSize(2 * dim).dropout(dropout)
                    .create(Shape.of(1, dim), r));
        }
        this.head = AttentionModel.matrix(r, dim, 2 * harmonics);
    }

    /**
     * Creates the model.
     *
     * @param dim encoder width
     * @param layers encoder blocks
     * @param harmonics Fourier harmonics of the position score
     * @param dropout encoder dropout (used for MC-dropout candidates)
     * @param seed the seed
     * @return the model
     */
    public static SoftPermutationTsp of(int dim, int layers, int harmonics, float dropout, long seed) {
        return new SoftPermutationTsp(dim, layers, harmonics, dropout, seed);
    }

    /**
     * @param tau the Gumbel-Sinkhorn temperature of training
     * @return this
     */
    public SoftPermutationTsp temperature(float tau) {
        this.temperature = tau;
        return this;
    }

    private Tensor basis(int n) {
        float[] p = new float[2 * harmonics * n];
        for (int h = 0; h < harmonics; h++) {
            for (int k = 0; k < n; k++) {
                p[(2 * h) * n + k] = (float) Math.cos(2 * Math.PI * (h + 1) * k / n);
                p[(2 * h + 1) * n + k] = (float) Math.sin(2 * Math.PI * (h + 1) * k / n);
            }
        }
        return Tensor.of(Shape.of(2 * harmonics, n), p);
    }

    /**
     * Position scores.
     *
     * @param envs same-size instances
     * @param training dropout on
     * @return {@code [batch, n, n]}: node by position
     */
    public Var scores(List<RoutingEnv> envs, boolean training) {
        int b = envs.size();
        int n = envs.get(0).nodes();
        float[] x = new float[b * n * 2];
        for (int s = 0; s < b; s++) {
            double[][] c = envs.get(s).coords();
            for (int i = 0; i < n; i++) {
                x[(s * n + i) * 2] = (float) (c[i][0] - 0.5);
                x[(s * n + i) * 2 + 1] = (float) (c[i][1] - 0.5);
            }
        }
        Var h = Var.of(Tensor.of(Shape.of(b, n, 2), x)).matmul(embed).add(embedBias);
        for (TransformerBlock block : blocks) {
            h = block.forward(List.of(h), training);
        }
        return h.matmul(head).matmul(basis(n));
    }

    /**
     * The unsupervised loss on a batch: expected tour length under the Gumbel-Sinkhorn soft
     * permutation, divided by {@code sqrt(n)}.
     *
     * @param envs same-size instances
     * @param random the Gumbel source
     * @return the loss
     */
    public Var loss(List<RoutingEnv> envs, SplittableRandom random) {
        int b = envs.size();
        int n = envs.get(0).nodes();
        Var s = scores(envs, true);
        float[] g = new float[b * n * n];
        float[] d = new float[b * n * n];
        for (int q = 0; q < b; q++) {
            double[][] dist = envs.get(q).distances();
            double mean = 0;
            for (double[] row : dist) {
                for (double v : row) {
                    mean += v;
                }
            }
            mean /= (double) n * (n - 1);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    d[(q * n + i) * n + j] = (float) (i == j ? 2 * mean : dist[i][j]);
                }
            }
        }
        for (int i = 0; i < g.length; i++) {
            g[i] = (float) -Math.log(-Math.log(Math.max(1e-9, random.nextDouble())));
        }
        Var logP = s.add(Tensor.of(Shape.of(b, n, n), g).mul(0.3f)).mul(1 / temperature);
        for (int it = 0; it < sinkhornIterations; it++) {
            logP = logP.sub(logP.logsumexp(2, true));
            logP = logP.sub(logP.logsumexp(1, true));
        }
        Var p = logP.exp();
        Var next = Var.concat(2, p.slice(2, 1, n), p.slice(2, 0, 1));
        Var length = p.mul(Var.of(Tensor.of(Shape.of(b, n, n), d)).matmul(next)).sum();
        return length.mul((float) (1 / (b * Math.sqrt(n))));
    }

    /**
     * Trains on fresh random instances.
     *
     * @param n nodes
     * @param steps steps
     * @param batch instances per step
     * @param optimizer the optimizer
     * @param random the source
     * @return the last loss
     */
    public float train(int n, int steps, int batch, Optimizer optimizer, SplittableRandom random) {
        float last = 0;
        for (int s = 0; s < steps; s++) {
            List<RoutingEnv> envs = new ArrayList<>();
            for (int i = 0; i < batch; i++) {
                envs.add(RoutingEnv.tsp(io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.points(n,
                        false, random)));
            }
            Var loss = loss(envs, random);
            last = loss.value().item();
            AttentionModel.step(loss, parameters(), optimizer, 1.0);
        }
        return last;
    }

    /**
     * Decodes tours: the first from the deterministic scores, the rest from MC dropout and
     * Gumbel noise; returns the shortest.
     *
     * @param env the instance
     * @param candidates how many tours to try
     * @param random the noise source
     * @return the best tour
     */
    public int[] decode(RoutingEnv env, int candidates, SplittableRandom random) {
        int n = env.nodes();
        int[] best = null;
        double bestLen = Double.POSITIVE_INFINITY;
        for (int c = 0; c < candidates; c++) {
            boolean noisy = c > 0;
            float[] s = Var.noGrad(() -> scores(List.of(env), noisy)).value().toArray();
            double[][] cost = new double[n][n];
            double max = 0;
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    cost[i][k] = -s[i * n + k];
                    max = Math.max(max, Math.abs(cost[i][k]));
                }
            }
            EntropicTransport.Plan plan = EntropicTransport.sinkhorn(EntropicTransport.uniform(n),
                    EntropicTransport.uniform(n), cost, Math.max(1e-3, 0.05 * max), 200, 1e-6);
            double[][] coupling = plan.coupling();
            double[][] assign = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    double gumbel = noisy ? -Math.log(-Math.log(Math.max(1e-12, random.nextDouble()))) : 0;
                    assign[i][k] = -Math.log(coupling[i][k] * n + 1e-30) - 0.5 * gumbel;
                }
            }
            int[] pos = MinibatchOptimalTransport.assignCost(assign);
            int[] tour = new int[n];
            for (int i = 0; i < n; i++) {
                tour[pos[i]] = i;
            }
            double len = env.length(tour);
            if (len < bestLen) {
                bestLen = len;
                best = tour;
            }
        }
        return best;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, embedBias));
        for (TransformerBlock block : blocks) {
            p.addAll(block.parameters());
        }
        p.add(head);
        return p;
    }
}
