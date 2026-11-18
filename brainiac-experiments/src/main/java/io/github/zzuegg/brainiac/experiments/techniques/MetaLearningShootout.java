package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.FewShotEpisode;
import io.github.zzuegg.brainiac.experimental.nn.Cnaps;
import io.github.zzuegg.brainiac.experimental.nn.MatchingNetwork;
import io.github.zzuegg.brainiac.experimental.nn.PrototypicalNetworks;
import io.github.zzuegg.brainiac.experimental.nn.SnailLearner;
import io.github.zzuegg.brainiac.experimental.train.LatentEmbeddingOptimization;
import io.github.zzuegg.brainiac.experimental.train.Maml;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Few-shot classification shootout of meta-learners on {@link FewShotEpisodes}: 2-way
 * episodes, meta-trained on some classes and tested on 100 episodes of classes never seen.
 * Every learner gets the same episode stream and a comparable budget.
 *
 * <ul>
 *   <li>{@link Method#PROTOTYPICAL} and {@link Method#MATCHING}: metric learners (no inner loop).</li>
 *   <li>{@link Method#MAML}: second-order MAML over the whole MLP, 3 inner steps.</li>
 *   <li>{@link Method#ANIL}: the same, but only the head adapts.</li>
 *   <li>{@link Method#META_SGD}: MAML with learned per-weight inner rates.</li>
 *   <li>{@link Method#SNAIL}: the episode as a sequence, no inner loop.</li>
 *   <li>{@link Method#LEO}: latent-space adaptation on features of an extractor pretrained on
 *       the training classes.</li>
 *   <li>{@link Method#CNAPS}: FiLM-adapted extractor plus amortized head, no inner loop.</li>
 * </ul>
 */
@SuppressWarnings("exports")
public final class MetaLearningShootout {

    /** The meta-learners. */
    public enum Method {
        /** Prototypical networks. */
        PROTOTYPICAL,
        /** Matching networks. */
        MATCHING,
        /** Second-order MAML. */
        MAML,
        /** MAML adapting only the head. */
        ANIL,
        /** MAML with learned per-weight inner rates. */
        META_SGD,
        /** Simple neural attentive learner. */
        SNAIL,
        /** Latent embedding optimization. */
        LEO,
        /** Conditional neural adaptive processes. */
        CNAPS
    }

    /**
     * One measurement.
     *
     * @param accuracy mean query accuracy over the test episodes
     * @param millisPerEpisode wall time to adapt to and classify one test episode
     * @param trainMillis wall time of meta-training
     * @param adaptedWeights numbers changed per task by an inner loop (0 for amortized learners)
     */
    public record Result(double accuracy, double millisPerEpisode, long trainMillis, long adaptedWeights) {
    }

    private static final int WAYS = 2;
    private static final int QUERIES = 5;
    private static final int TEST_EPISODES = 100;
    private static final int INNER = 3;

    private MetaLearningShootout() {
    }

    /**
     * Meta-trains one learner and tests it.
     *
     * @param method the learner
     * @param source the episodes
     * @param shots support examples per class
     * @param episodes meta-training episodes
     * @param seed the seed
     * @return the result
     */
    public static Result run(Method method, FewShotEpisodes source, int shots, int episodes, long seed) {
        return switch (method) {
            case PROTOTYPICAL, MATCHING -> metric(method, source, shots, episodes, seed);
            case MAML, ANIL, META_SGD -> maml(method, source, shots, episodes, seed, INNER);
            case SNAIL -> snail(source, shots, episodes, seed);
            case LEO -> leo(source, shots, episodes, seed, INNER);
            case CNAPS -> cnaps(source, shots, episodes, seed, true);
        };
    }

    private interface Classifier {
        int[] classify(FewShotEpisode e);
    }

    private static double[] test(FewShotEpisodes source, int shots, long seed, Classifier c) {
        SplittableRandom r = new SplittableRandom(seed + 999);
        double acc = 0;
        long t0 = System.nanoTime();
        for (int e = 0; e < TEST_EPISODES; e++) {
            FewShotEpisode ep = source.test(WAYS, shots, QUERIES, r);
            acc += ep.accuracy(c.classify(ep));
        }
        return new double[] {acc / TEST_EPISODES, (System.nanoTime() - t0) / 1e6 / TEST_EPISODES};
    }

    private static Model embedding(long seed) {
        return Model.sequential().input(FewShotEpisodes.FEATURES).dense(64, Activation.RELU)
                .dense(32, Activation.IDENTITY).seed(seed).build();
    }

    private static Result metric(Method method, FewShotEpisodes source, int shots, int episodes, long seed) {
        Model embed = embedding(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        long t0 = System.currentTimeMillis();
        for (int e = 0; e < episodes; e++) {
            FewShotEpisode ep = source.train(WAYS, shots, QUERIES, r);
            Var s = embed.forward(Var.of(ep.support()));
            Var q = embed.forward(Var.of(ep.query()));
            Var loss = method == Method.MATCHING
                    ? MatchingNetwork.loss(s, ep.supportLabels(), q, ep.queryLabels(), WAYS, 10f)
                    : PrototypicalNetworks.loss(s, ep.supportLabels(), q, ep.queryLabels(), WAYS);
            loss.backward();
            opt.step(embed.parameters());
        }
        long train = System.currentTimeMillis() - t0;
        double[] t = test(source, shots, seed, ep -> {
            Var s = embed.forward(Var.of(ep.support()));
            Var q = embed.forward(Var.of(ep.query()));
            return method == Method.MATCHING ? MatchingNetwork.classify(q, s, ep.supportLabels(), WAYS, 10f)
                    : PrototypicalNetworks.classify(q, PrototypicalNetworks.prototypes(s, ep.supportLabels(), WAYS));
        });
        return new Result(t[0], t[1], train, 0);
    }

    /** A MAML task over one episode, scored with the model's current weights. */
    private record EpisodeTask(Model model, FewShotEpisode ep) implements Maml.Task {

        @Override
        public Var supportLoss() {
            return Functional.crossEntropy(model.forward(Var.of(ep.support())), ep.supportIds());
        }

        @Override
        public Var queryLoss() {
            return Functional.crossEntropy(model.forward(Var.of(ep.query())), ep.queryIds());
        }
    }

    /**
     * MAML, ANIL or Meta-SGD, also scoring the learner when it takes more inner steps at test
     * time than it was trained with.
     *
     * @param method {@link Method#MAML}, {@link Method#ANIL} or {@link Method#META_SGD}
     * @param source the episodes
     * @param shots support examples per class
     * @param metaSteps meta steps of 4 episodes each
     * @param seed the seed
     * @param testSteps inner steps at test time
     * @return the result
     */
    public static Result maml(Method method, FewShotEpisodes source, int shots, int metaSteps, long seed,
            int testSteps) {
        Model model = Model.sequential().input(FewShotEpisodes.FEATURES).dense(32, Activation.RELU, Init.XAVIER)
                .dense(32, Activation.RELU, Init.XAVIER).dense(WAYS, Activation.IDENTITY).seed(seed).build();
        Maml maml = new Maml(model.parameters(), Optimizer.adam(2e-3f), 0.3f, INNER, true);
        if (method == Method.ANIL) {
            List<io.github.zzuegg.brainiac.nn.Layer> layers = model.layers();
            maml.anil(layers.get(layers.size() - 1).parameters());
        } else if (method == Method.META_SGD) {
            maml.metaSgd();
        }
        SplittableRandom r = new SplittableRandom(seed);
        long t0 = System.currentTimeMillis();
        for (int it = 0; it < metaSteps; it++) {
            List<EpisodeTask> tasks = new ArrayList<>();
            for (int k = 0; k < 4; k++) {
                tasks.add(new EpisodeTask(model, source.train(WAYS, shots, QUERIES, r)));
            }
            maml.metaStep(tasks);
        }
        long train = System.currentTimeMillis() - t0;
        double[] t = test(source, shots, seed, ep -> {
            List<Tensor> start = maml.snapshot();
            maml.adapt(new EpisodeTask(model, ep), testSteps);
            int[] pred = FewShotEpisode.argmax(model.forward(Var.of(ep.query())).value());
            maml.restore(start);
            return pred;
        });
        return new Result(t[0], t[1], train, maml.adaptedWeights());
    }

    /** The episode as SNAIL sequences: support (x, one-hot y) then the query (x, 0). */
    private static Tensor sequences(FewShotEpisode ep) {
        int n = ep.supportLabels().length;
        int m = ep.queryLabels().length;
        int f = ep.features();
        int w = f + ep.ways();
        float[] s = ep.support().toArray();
        float[] q = ep.query().toArray();
        float[] out = new float[m * (n + 1) * w];
        for (int b = 0; b < m; b++) {
            for (int i = 0; i < n; i++) {
                int o = (b * (n + 1) + i) * w;
                System.arraycopy(s, i * f, out, o, f);
                out[o + f + ep.supportLabels()[i]] = 1;
            }
            System.arraycopy(q, b * f, out, (b * (n + 1) + n) * w, f);
        }
        return Tensor.of(Shape.of(m, n + 1, w), out);
    }

    private static Result snail(FewShotEpisodes source, int shots, int episodes, long seed) {
        int length = WAYS * shots + 1;
        SnailLearner snail = new SnailLearner(FewShotEpisodes.FEATURES + WAYS, WAYS, length, 16, 16, 16,
                new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        long t0 = System.currentTimeMillis();
        for (int e = 0; e < episodes; e++) {
            FewShotEpisode ep = source.train(WAYS, shots, QUERIES, r);
            Functional.crossEntropy(snail.last(Var.of(sequences(ep))), ep.queryIds()).backward();
            opt.step(snail.parameters());
        }
        long train = System.currentTimeMillis() - t0;
        double[] t = test(source, shots, seed,
                ep -> FewShotEpisode.argmax(snail.last(Var.of(sequences(ep))).value()));
        return new Result(t[0], t[1], train, 0);
    }

    /**
     * LEO, also scoring it with more latent inner steps at test time than in training.
     *
     * @param source the episodes
     * @param shots support examples per class
     * @param metaSteps meta steps of 4 episodes each
     * @param seed the seed
     * @param testSteps latent inner steps at test time
     * @return the result
     */
    public static Result leo(FewShotEpisodes source, int shots, int metaSteps, long seed, int testSteps) {
        SplittableRandom r = new SplittableRandom(seed);
        long t0 = System.currentTimeMillis();
        // the paper's extractor is pretrained by plain classification on the training classes
        ToyTasks.Data pre = source.pretraining(150, r);
        Model net = Model.sequential().input(FewShotEpisodes.FEATURES).dense(64, Activation.RELU)
                .dense(32, Activation.RELU).dense(pre.classes(), Activation.IDENTITY).seed(seed).build();
        Optimizer popt = Optimizer.adam(3e-3f);
        for (int epoch = 0; epoch < 60; epoch++) {
            Functional.crossEntropy(net.forward(Var.of(pre.inputs())), pre.ids()).backward();
            popt.step(net.parameters());
        }
        Dense l1 = (Dense) net.layers().get(0);
        Dense l2 = (Dense) net.layers().get(1);
        java.util.function.UnaryOperator<Tensor> feats = x -> l2.forward(l1.forward(Var.of(x))).value();
        LatentEmbeddingOptimization leo = new LatentEmbeddingOptimization(32, 8, 32, 5, 0,
                new SplittableRandom(seed + 1));
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int it = 0; it < metaSteps; it++) {
            List<FewShotEpisode> batch = new ArrayList<>();
            for (int k = 0; k < 4; k++) {
                batch.add(featurize(source.train(WAYS, shots, QUERIES, r), feats));
            }
            leo.metaStep(batch, opt);
        }
        long train = System.currentTimeMillis() - t0;
        double[] t = test(source, shots, seed, ep -> leo.classify(featurize(ep, feats), testSteps));
        return new Result(t[0], t[1], train, (long) WAYS * leo.latent());
    }

    private static FewShotEpisode featurize(FewShotEpisode ep, java.util.function.UnaryOperator<Tensor> f) {
        return new FewShotEpisode(f.apply(ep.support()), ep.supportLabels(), f.apply(ep.query()), ep.queryLabels(),
                ep.ways());
    }

    /**
     * CNAPs, with or without FiLM adaptation.
     *
     * @param source the episodes
     * @param shots support examples per class
     * @param episodes training episodes
     * @param seed the seed
     * @param adaptive whether the extractor is FiLM-adapted
     * @return the result
     */
    public static Result cnaps(FewShotEpisodes source, int shots, int episodes, long seed, boolean adaptive) {
        Cnaps model = new Cnaps(FewShotEpisodes.FEATURES, 64, 16, adaptive, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        long t0 = System.currentTimeMillis();
        for (int e = 0; e < episodes; e++) {
            model.loss(source.train(WAYS, shots, QUERIES, r)).backward();
            opt.step(model.parameters());
        }
        long train = System.currentTimeMillis() - t0;
        double[] t = test(source, shots, seed, model::classify);
        return new Result(t[0], t[1], train, 0);
    }
}
