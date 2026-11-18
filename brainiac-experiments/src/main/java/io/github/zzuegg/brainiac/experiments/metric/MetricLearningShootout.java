package io.github.zzuegg.brainiac.experiments.metric;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.metric.MarginSoftmax;
import io.github.zzuegg.brainiac.experimental.metric.PairLosses;
import io.github.zzuegg.brainiac.experimental.metric.ProxyLosses;
import io.github.zzuegg.brainiac.experimental.metric.RetrievalMetrics;
import io.github.zzuegg.brainiac.experimental.metric.SupervisedContrastive;
import io.github.zzuegg.brainiac.experimental.ml.ClusteringMetrics;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.nn.PrototypicalNetworks;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.experimental.ssl.KoLeo;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.experiments.multimodal.DigitGlyphs;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 10 items 10.3.7-10.3.10 (and 10.3.5's KoLeo as a regulariser): deep metric learning
 * losses under one fair protocol (Musgrave et al., ECCV 2020): the same MLP embedding network
 * (64 to 128 to 32), the same class-balanced batches (8 images of each of 6 classes), the same
 * augmentation (pixel noise, flips and one-pixel shifts), the same steps and Adam settings
 * (loss parameters such as proxies and class centres get a 10x learning rate, as the papers
 * do).
 *
 * <p>Task: {@link DigitGlyphs}, ten 3x5 digit glyphs drawn at a random one of 24 positions in an
 * 8x8 image with 5% pixel flips, so raw pixels retrieve poorly and a useful embedding must
 * learn shape across positions. Training uses classes 0-5; the embeddings
 * are then judged on held-out images of the <em>unseen</em> classes 6-9 (retrieval Recall@1 and
 * {@literal @}4, NMI of k-means on the embeddings, verification equal error rate, intra/inter-class
 * angles), on held-out images of the seen classes (Recall@1) and on seen classes with three
 * times the pixel-flip noise (a corruption check).
 *
 * <p>Baselines: cross-entropy with the embedding taken before the linear classifier, SimCLR
 * (no labels) and prototypical networks (episodic), next to SupCon, contrastive, triplet with
 * random, hard and semi-hard mining, multi-similarity, normalised softmax, CosFace, ArcFace,
 * sub-center ArcFace, Proxy-NCA, Proxy-NCA++ and Proxy-Anchor.
 */
public final class MetricLearningShootout {

    /** The training objective. */
    public enum Objective {
        /** Softmax cross-entropy through a linear head (embedding before it). */
        CROSS_ENTROPY,
        /** SimCLR's NT-Xent, labels unused. */
        SIMCLR,
        /** Prototypical networks on episodes cut from the batch. */
        PROTOTYPICAL,
        /** Supervised contrastive. */
        SUPCON,
        /** Supervised contrastive plus KoLeo (weight 0.1). */
        SUPCON_KOLEO,
        /** Contrastive pairs. */
        CONTRASTIVE,
        /** Triplet with random mining. */
        TRIPLET_RANDOM,
        /** Triplet with batch-hard mining. */
        TRIPLET_HARD,
        /** Triplet with semi-hard mining. */
        TRIPLET_SEMI_HARD,
        /** Multi-similarity. */
        MULTI_SIMILARITY,
        /** Normalised softmax (no margin). */
        NORM_SOFTMAX,
        /** CosFace. */
        COSFACE,
        /** ArcFace. */
        ARCFACE,
        /** Sub-center ArcFace (3 centres per class). */
        SUBCENTER_ARCFACE,
        /** Proxy-NCA. */
        PROXY_NCA,
        /** Proxy-NCA++. */
        PROXY_NCA_PLUS_PLUS,
        /** Proxy-Anchor. */
        PROXY_ANCHOR
    }

    /**
     * Scores of one objective.
     *
     * @param objective the objective
     * @param recall1 Recall@1 on unseen classes
     * @param recall4 Recall@4 on unseen classes
     * @param nmi NMI of 4-means on the unseen-class embeddings
     * @param eer verification equal error rate on unseen classes
     * @param seenRecall1 Recall@1 on held-out images of the training classes
     * @param corruptedRecall1 the same with three times the pixel noise
     * @param intraAngle mean same-class angle on unseen classes, degrees
     * @param interAngle mean different-class angle on unseen classes, degrees
     * @param millisPerStep the training wall-clock per step
     */
    public record Scores(Objective objective, double recall1, double recall4, double nmi, double eer,
            double seenRecall1, double corruptedRecall1, double intraAngle, double interAngle, double millisPerStep) {

        @Override
        public String toString() {
            return String.format("%s R@1 %.3f R@4 %.3f NMI %.3f EER %.3f seen R@1 %.3f corrupted %.3f angles %.0f/%.0f"
                    + " %.2f ms/step", objective, recall1, recall4, nmi, eer, seenRecall1, corruptedRecall1,
                    intraAngle, interAngle, millisPerStep);
        }
    }

    private static final int SEEN = 6;
    private static final int PER_CLASS = 8;
    private static final int DIM = 32;

    private MetricLearningShootout() {
    }

    private static ToyTasks.Data all(long sampleSeed, int perClass, double flip) {
        SplittableRandom r = new SplittableRandom(sampleSeed);
        float[][] x = new float[10 * perClass][];
        int[] y = new int[10 * perClass];
        for (int i = 0; i < x.length; i++) {
            y[i] = i % 10;
            x[i] = DigitGlyphs.sample(y[i], flip, r);
        }
        return new ToyTasks.Data(x, y, 10);
    }

    private static Tensor augment(float[][] x, int[] idx, SplittableRandom r) {
        int d = x[0].length;
        int side = 8;
        float[] out = new float[idx.length * d];
        for (int i = 0; i < idx.length; i++) {
            int dx = r.nextInt(3) - 1;
            int dy = r.nextInt(3) - 1;
            for (int py = 0; py < side; py++) {
                for (int px = 0; px < side; px++) {
                    int sy = Math.floorMod(py - dy, side);
                    int sx = Math.floorMod(px - dx, side);
                    float v = x[idx[i]][sy * side + sx];
                    if (r.nextDouble() < 0.05) {
                        v = -v;
                    }
                    out[i * d + py * side + px] = v + (float) (r.nextGaussian() * 0.1);
                }
            }
        }
        return Tensor.of(Shape.of(idx.length, d), out);
    }

    /** A class-balanced batch: PER_CLASS indices of each seen class. */
    private static int[] balanced(int[][] byClass, SplittableRandom r) {
        int[] idx = new int[SEEN * PER_CLASS];
        for (int c = 0; c < SEEN; c++) {
            for (int k = 0; k < PER_CLASS; k++) {
                idx[c * PER_CLASS + k] = byClass[c][r.nextInt(byClass[c].length)];
            }
        }
        return idx;
    }

    private static float[][] embed(Model encoder, float[][] x) {
        Tensor e = Var.noGrad(() -> encoder.forward(Var.of(Tensor.of(x)))).value();
        float[] v = e.toArray();
        float[][] out = new float[x.length][DIM];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(v, i * DIM, out[i], 0, DIM);
        }
        return out;
    }

    /**
     * Trains one objective and scores it.
     *
     * @param objective the objective
     * @param steps training steps
     * @param seed the seed
     * @return the scores
     */
    public static Scores run(Objective objective, int steps, long seed) {
        return curve(objective, new int[] {steps}, seed).get(0);
    }

    /**
     * Trains one objective and scores it at several step counts (convergence speed).
     *
     * @param objective the objective
     * @param checkpoints ascending step counts
     * @param seed the seed
     * @return one score per checkpoint
     */
    public static List<Scores> curve(Objective objective, int[] checkpoints, long seed) {
        ToyTasks.Data train = all(seed * 31 + 7, 60, 0.05).onlyClasses(0, 1, 2, 3, 4, 5);
        ToyTasks.Data unseen = all(seed * 31 + 9, 40, 0.05).onlyClasses(6, 7, 8, 9);
        ToyTasks.Data seen = all(seed * 31 + 11, 30, 0.05).onlyClasses(0, 1, 2, 3, 4, 5);
        ToyTasks.Data corrupted = all(seed * 31 + 13, 30, 0.15).onlyClasses(0, 1, 2, 3, 4, 5);
        int[][] byClass = new int[SEEN][];
        for (int c = 0; c < SEEN; c++) {
            int cls = c;
            byClass[c] = java.util.stream.IntStream.range(0, train.size()).filter(i -> train.y()[i] == cls).toArray();
        }
        Model encoder = Model.sequential().input(64).dense(128, Activation.RELU).dense(DIM, Activation.IDENTITY)
                .seed(seed).build();
        Model head = Model.sequential().input(DIM).dense(SEEN, Activation.IDENTITY).seed(seed + 1).build();
        Model projector = Model.sequential().input(DIM).dense(DIM, Activation.RELU).dense(DIM, Activation.IDENTITY)
                .seed(seed + 2).build();
        List<Var> lossParams = new ArrayList<>();
        Loss loss = switch (objective) {
            case SUPCON, SUPCON_KOLEO -> new SupervisedContrastive(0.1f);
            case CONTRASTIVE -> PairLosses.contrastive(1f);
            case TRIPLET_RANDOM -> PairLosses.triplet(0.2f, PairLosses.Mining.RANDOM, seed);
            case TRIPLET_HARD -> PairLosses.triplet(0.2f, PairLosses.Mining.HARD, seed);
            case TRIPLET_SEMI_HARD -> PairLosses.triplet(0.2f, PairLosses.Mining.SEMI_HARD, seed);
            case MULTI_SIMILARITY -> PairLosses.multiSimilarity(2, 50, 0.5f, 0.1f);
            case NORM_SOFTMAX, COSFACE, ARCFACE, SUBCENTER_ARCFACE -> {
                MarginSoftmax m = switch (objective) {
                    case NORM_SOFTMAX -> MarginSoftmax.normalized(DIM, SEEN, seed + 3);
                    case COSFACE -> MarginSoftmax.cosFace(DIM, SEEN, seed + 3);
                    case ARCFACE -> MarginSoftmax.arcFace(DIM, SEEN, seed + 3);
                    default -> new MarginSoftmax(MarginSoftmax.Kind.ARCFACE, DIM, SEEN, 16, 0.3f, 3, false, seed + 3);
                };
                lossParams.addAll(m.parameters());
                yield m;
            }
            case PROXY_NCA, PROXY_NCA_PLUS_PLUS, PROXY_ANCHOR -> {
                ProxyLosses p = new ProxyLosses(switch (objective) {
                    case PROXY_NCA -> ProxyLosses.Kind.PROXY_NCA;
                    case PROXY_NCA_PLUS_PLUS -> ProxyLosses.Kind.PROXY_NCA_PLUS_PLUS;
                    default -> ProxyLosses.Kind.PROXY_ANCHOR;
                }, DIM, SEEN, seed + 3);
                lossParams.addAll(p.parameters());
                yield p;
            }
            default -> null;
        };
        List<Var> params = new ArrayList<>(encoder.parameters());
        if (objective == Objective.CROSS_ENTROPY) {
            params.addAll(head.parameters());
        }
        if (objective == Objective.SIMCLR) {
            params.addAll(projector.parameters());
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        Optimizer fast = Optimizer.adam(3e-2f);
        SplittableRandom r = new SplittableRandom(seed + 4);
        List<Scores> out = new ArrayList<>();
        int done = 0;
        long elapsed = 0;
        for (int checkpoint : checkpoints) {
            long start = System.nanoTime();
            for (; done < checkpoint; done++) {
                int[] idx = balanced(byClass, r);
                float[] ids = new float[idx.length];
                for (int i = 0; i < idx.length; i++) {
                    ids[i] = train.y()[idx[i]];
                }
                Tensor labels = Tensor.of(Shape.of(idx.length), ids);
                Var a = encoder.forward(Var.of(augment(train.x(), idx, r)));
                Var l = switch (objective) {
                    case CROSS_ENTROPY -> Functional.crossEntropy(head.forward(a), labels);
                    case SIMCLR -> SelfSupervisedLosses.ntXent(projector.forward(a),
                            projector.forward(encoder.forward(Var.of(augment(train.x(), idx, r)))), 0.2f);
                    case PROTOTYPICAL -> prototypical(a);
                    case SUPCON, SUPCON_KOLEO -> {
                        Var b = encoder.forward(Var.of(augment(train.x(), idx, r)));
                        Var sc = loss.compute(Var.concat(0, a, b), Tensor.concat(0, labels, labels));
                        yield objective == Objective.SUPCON_KOLEO ? sc.add(KoLeo.loss(a).mul(0.1f)) : sc;
                    }
                    default -> loss.compute(a, labels);
                };
                l.backward();
                opt.step(params);
                if (!lossParams.isEmpty()) {
                    fast.step(lossParams);
                }
            }
            elapsed += System.nanoTime() - start;
            out.add(score(objective, encoder, unseen, seen, corrupted, elapsed / 1e6 / Math.max(1, done), seed));
        }
        return out;
    }

    /** Episodes cut from the balanced batch: 4 support and 4 query images per class. */
    private static Var prototypical(Var embeddings) {
        int half = PER_CLASS / 2;
        int[] supportRows = new int[SEEN * half];
        int[] queryRows = new int[SEEN * half];
        int[] supportLabels = new int[SEEN * half];
        float[] queryLabels = new float[SEEN * half];
        for (int c = 0; c < SEEN; c++) {
            for (int k = 0; k < half; k++) {
                supportRows[c * half + k] = c * PER_CLASS + k;
                queryRows[c * half + k] = c * PER_CLASS + half + k;
                supportLabels[c * half + k] = c;
                queryLabels[c * half + k] = c;
            }
        }
        Var support = embeddings.gather(0, ids(supportRows));
        Var query = embeddings.gather(0, ids(queryRows));
        Var logits = PrototypicalNetworks.logits(query, PrototypicalNetworks.prototypes(support, supportLabels, SEEN));
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(queryLabels.length), queryLabels));
    }

    private static Tensor ids(int[] rows) {
        float[] v = new float[rows.length];
        for (int i = 0; i < rows.length; i++) {
            v[i] = rows[i];
        }
        return Tensor.of(Shape.of(rows.length), v);
    }

    private static Scores score(Objective objective, Model encoder, ToyTasks.Data unseen, ToyTasks.Data seen,
            ToyTasks.Data corrupted, double millis, long seed) {
        float[][] e = embed(encoder, unseen.x());
        double[][] unit = new double[e.length][DIM];
        for (int i = 0; i < e.length; i++) {
            double s = 1e-12;
            for (float v : e[i]) {
                s += (double) v * v;
            }
            s = Math.sqrt(s);
            for (int j = 0; j < DIM; j++) {
                unit[i][j] = e[i][j] / s;
            }
        }
        int[] clusters = KMeans.fit(unit, 4, KMeans.Init.PLUS_PLUS, 5, 100, seed).labels();
        int[] truth = Arrays.stream(unseen.y()).map(y -> y - SEEN).toArray();
        double[] angles = RetrievalMetrics.angles(e, unseen.y());
        return new Scores(objective, RetrievalMetrics.recallAt(e, unseen.y(), 1),
                RetrievalMetrics.recallAt(e, unseen.y(), 4), ClusteringMetrics.nmi(truth, clusters),
                RetrievalMetrics.equalErrorRate(e, unseen.y()), RetrievalMetrics.recallAt(embed(encoder, seen.x()),
                        seen.y(), 1), RetrievalMetrics.recallAt(embed(encoder, corrupted.x()), corrupted.y(), 1),
                angles[0], angles[1], millis);
    }

    /**
     * The untrained encoder's scores, the floor every objective must beat.
     *
     * @param seed the seed
     * @return the scores (objective field CROSS_ENTROPY at zero steps)
     */
    public static Scores untrained(long seed) {
        return run(Objective.CROSS_ENTROPY, 0, seed);
    }

    /**
     * The long-budget comparison (a Run): every objective at 1500 steps, three seeds, and
     * Recall@1 curves for the proxy and pair losses.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (long seed = 1; seed <= 3; seed++) {
            System.out.println("untrained " + untrained(seed));
            for (Objective o : Objective.values()) {
                System.out.println(run(o, 1500, seed));
            }
        }
        int[] checkpoints = {50, 100, 200, 400, 800, 1500};
        for (Objective o : List.of(Objective.PROXY_ANCHOR, Objective.PROXY_NCA, Objective.MULTI_SIMILARITY,
                Objective.TRIPLET_SEMI_HARD, Objective.SUPCON)) {
            StringBuilder line = new StringBuilder(o + " R@1 curve:");
            for (Scores s : curve(o, checkpoints, 1)) {
                line.append(String.format(" %.3f", s.recall1()));
            }
            System.out.println(line);
        }
    }
}
