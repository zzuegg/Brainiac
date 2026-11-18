package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.FewShotEpisode;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * LEO (latent embedding optimization): MAML-style few-shot adaptation that runs its inner
 * loop in a small latent space instead of in the high-dimensional weight space.
 *
 * <ol>
 *   <li>An <b>encoder</b> embeds every support example; a <b>relation network</b> looks at
 *       every pair of support examples and, averaged over the pairs that start in class
 *       {@code c}, gives a Gaussian over that class's latent code {@code z_c}.</li>
 *   <li>A <b>decoder</b> maps {@code z_c} to a Gaussian over the class's linear classifier
 *       weights {@code w_c} (a stochastic, data-dependent initialization).</li>
 *   <li>The <b>inner loop</b> takes gradient steps on the support loss with respect to the
 *       codes {@code z} (a learned step size), then optionally a few steps directly on the
 *       decoded weights {@code w} ("fine-tuning").</li>
 *   <li>The <b>outer loop</b> trains encoder, relation net, decoder and both step sizes on
 *       the query loss after adaptation, differentiating through the inner loop, plus a KL
 *       term pulling the codes toward {@code N(0, I)} and a penalty keeping the encoder's
 *       code close to the adapted one.</li>
 * </ol>
 *
 * <p>Because the inner loop moves only {@code ways * latent} numbers, it is cheap and
 * stable: taking more inner steps at test time than in training does not blow up. As in
 * the paper, it works on fixed features from a pretrained extractor; the episodes given to
 * this class already hold those features.
 *
 * <p>Paper: Rusu, Rao, Sygnowski, Vinyals, Pascanu, Osindero, Hadsell, "Meta-Learning with
 * Latent Embedding Optimization" (2019), <a href="https://arxiv.org/abs/1807.05960">arXiv:1807.05960</a>.
 */
public final class LatentEmbeddingOptimization {

    private final int features;
    private final int latent;
    private final int hidden;
    private final int innerSteps;
    private final int finetuneSteps;
    private final float klWeight;
    private final float encoderPenalty;
    private final SplittableRandom noise;
    private final Var encW;
    private final Var encB;
    private final Var relW1;
    private final Var relB1;
    private final Var relW2;
    private final Var relB2;
    private final Var decW;
    private final Var decB;
    private final Var innerRate;
    private final Var finetuneRate;

    /**
     * Creates the meta-learner.
     *
     * @param features feature size of each example
     * @param latent the size of each class code {@code z_c}
     * @param hidden the encoder width
     * @param innerSteps latent-space adaptation steps during meta-training
     * @param finetuneSteps weight-space fine-tuning steps after them (0 for none)
     * @param random the initializer, also seeds the sampling noise
     */
    public LatentEmbeddingOptimization(int features, int latent, int hidden, int innerSteps, int finetuneSteps,
            RandomGenerator random) {
        this.features = features;
        this.latent = latent;
        this.hidden = hidden;
        this.innerSteps = innerSteps;
        this.finetuneSteps = finetuneSteps;
        this.klWeight = 1e-3f;
        this.encoderPenalty = 1e-3f;
        this.noise = new SplittableRandom(random.nextLong());
        this.encW = weight(features, hidden, random);
        this.encB = Var.param(Tensor.zeros(Shape.of(hidden)));
        this.relW1 = weight(2 * hidden, 2 * hidden, random);
        this.relB1 = Var.param(Tensor.zeros(Shape.of(2 * hidden)));
        this.relW2 = weight(2 * hidden, 2 * latent, random);
        this.relB2 = Var.param(meanAndLogVar(latent));
        this.decW = weight(latent, 2 * features, random);
        this.decB = Var.param(meanAndLogVar(features));
        this.innerRate = Var.param(Tensor.full(Shape.of(1), 3f));
        this.finetuneRate = Var.param(Tensor.full(Shape.of(1), 0.01f));
    }

    /** A bias whose log-variance half starts low, so sampling starts almost deterministic. */
    private static Tensor meanAndLogVar(int n) {
        float[] b = new float[2 * n];
        java.util.Arrays.fill(b, n, 2 * n, -6f);
        return Tensor.of(Shape.of(2 * n), b);
    }

    private static Var weight(int in, int out, RandomGenerator random) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0f, (float) Math.sqrt(1.0 / in), random));
    }

    /** @return every trainable parameter, including the two learned step sizes */
    public List<Var> parameters() {
        return List.of(encW, encB, relW1, relB1, relW2, relB2, decW, decB, innerRate, finetuneRate);
    }

    /** The class-code Gaussians: {@code {mean, logVar}}, each {@code [ways, latent]}. */
    private Var[] encode(FewShotEpisode e) {
        int n = e.supportLabels().length;
        Var h = Var.of(e.support()).matmul(encW).add(encB).relu();
        float[] first = new float[n * n * n];
        float[] second = new float[n * n * n];
        float[] average = new float[e.ways() * n * n];
        int[] perClass = new int[e.ways()];
        for (int label : e.supportLabels()) {
            perClass[label]++;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int pair = i * n + j;
                first[pair * n + i] = 1;
                second[pair * n + j] = 1;
                average[e.supportLabels()[i] * n * n + pair] = 1f / (perClass[e.supportLabels()[i]] * n);
            }
        }
        Var pairs = Var.concat(1, Var.of(Tensor.of(Shape.of(n * n, n), first)).matmul(h),
                Var.of(Tensor.of(Shape.of(n * n, n), second)).matmul(h));
        Var r = pairs.matmul(relW1).add(relB1).relu().matmul(relW2).add(relB2);
        Var stats = Var.of(Tensor.of(Shape.of(e.ways(), n * n), average)).matmul(r);
        return new Var[] {stats.slice(1, 0, latent), stats.slice(1, latent, 2 * latent).clamp(-8f, 4f)};
    }

    private Var sample(Var mean, Var logVar, boolean stochastic) {
        if (!stochastic) {
            return mean;
        }
        Tensor eps = Tensor.normal(mean.shape(), 0f, 1f, noise);
        return mean.add(logVar.mul(0.5f).exp().mul(eps));
    }

    /** Classifier weights {@code [ways, features]} for codes {@code [ways, latent]}. */
    private Var decode(Var z, boolean stochastic) {
        Var d = z.matmul(decW).add(decB);
        return sample(d.slice(1, 0, features), d.slice(1, features, 2 * features).clamp(-8f, 4f), stochastic);
    }

    private static Var loss(Tensor x, Tensor ids, Var weights) {
        return Functional.crossEntropy(Var.of(x).matmul(weights.transpose()), ids);
    }

    /** The adapted weights and, for training, the regularizers. */
    private Var[] adapt(FewShotEpisode e, int steps, boolean training) {
        Var[] stats = encode(e);
        Var z0 = sample(stats[0], stats[1], training);
        Var z = z0;
        Tensor ids = e.supportIds();
        for (int s = 0; s < steps; s++) {
            Var g = Var.gradients(loss(e.support(), ids, decode(z, training)), List.of(z), training).get(0);
            z = z.sub(g.mul(innerRate));
            if (!training) {
                z = Var.param(z.value());
            }
        }
        Var w = training ? decode(z, true) : Var.param(decode(z, false).value());
        for (int s = 0; s < finetuneSteps; s++) {
            Var g = Var.gradients(loss(e.support(), ids, w), List.of(w), training).get(0);
            w = w.sub(g.mul(finetuneRate));
            if (!training) {
                w = Var.param(w.value());
            }
        }
        Var kl = stats[1].exp().add(stats[0].square()).sub(1f).sub(stats[1]).mul(0.5f).mean();
        Var drift = Var.of(z.value()).sub(z0).square().mean();
        return new Var[] {w, kl, drift};
    }

    /**
     * Runs one meta step over a batch of episodes.
     *
     * @param episodes the episodes (support and query hold features)
     * @param optimizer the outer optimizer
     * @return the mean query loss after adaptation
     */
    public float metaStep(List<FewShotEpisode> episodes, Optimizer optimizer) {
        Var total = null;
        double sum = 0;
        for (FewShotEpisode e : episodes) {
            Var[] a = adapt(e, innerSteps, true);
            Var q = loss(e.query(), e.queryIds(), a[0]);
            sum += q.value().item();
            Var l = q.add(a[1].mul(klWeight)).add(a[2].mul(encoderPenalty));
            total = total == null ? l : total.add(l);
        }
        for (Var p : parameters()) {
            p.zeroGrad();
        }
        total.div(episodes.size()).backward();
        optimizer.step(parameters());
        return (float) (sum / episodes.size());
    }

    /**
     * Adapts to an episode's support set and classifies its queries.
     *
     * @param episode the episode
     * @param steps latent adaptation steps, which may differ from training
     * @return the predicted class of each query
     */
    public int[] classify(FewShotEpisode episode, int steps) {
        Var w = adapt(episode, steps, false)[0];
        return FewShotEpisode.argmax(Var.of(episode.query()).matmul(w.transpose()).value());
    }

    /** @return the learned latent step size */
    public float innerRate() {
        return innerRate.value().item();
    }

    /** @return the latent size per class */
    public int latent() {
        return latent;
    }

    /** @return the encoder width */
    public int hidden() {
        return hidden;
    }
}
