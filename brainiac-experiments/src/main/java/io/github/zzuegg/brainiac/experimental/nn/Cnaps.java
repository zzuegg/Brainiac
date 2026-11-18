package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.FewShotEpisode;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * CNAPs (conditional neural adaptive processes): few-shot classification with no gradient
 * steps at test time. A <b>task encoder</b> summarizes the support set into one task
 * vector (a permutation-invariant mean of per-example codes); an <b>adaptation network</b>
 * turns it into {@link Functional#film FiLM} scales and shifts for every layer of a shared
 * feature extractor, so the features themselves change per task (useful when episodes come
 * from different domains); and an <b>amortized linear head</b> maps each class's mean
 * adapted feature to that class's weight vector and bias. Training is plain episodic
 * cross-entropy; adapting to a new task is one forward pass over its support set.
 *
 * <p>Simplified from the paper: the extractor is a two-layer MLP trained jointly (the paper
 * freezes a pretrained ResNet), and the head is a learned linear map of the class mean.
 * {@code adaptive = false} turns FiLM off, an ablation that leaves a prototypical-style
 * head on fixed features.
 *
 * <p>Paper: Requeima, Gordon, Bronskill, Nowozin, Turner, "Fast and Flexible Multi-Task
 * Classification Using Conditional Neural Adaptive Processes" (2019),
 * <a href="https://arxiv.org/abs/1906.07697">arXiv:1906.07697</a>.
 */
public final class Cnaps {

    private final int hidden;
    private final boolean adaptive;
    private final Var encW;
    private final Var encB;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Var film1;
    private final Var film2;
    private final Var headW;
    private final Var headB;

    /**
     * Creates the model.
     *
     * @param features input features per example
     * @param hidden the width of the feature extractor
     * @param task the size of the task vector
     * @param adaptive whether the extractor is FiLM-adapted per task
     * @param random the initializer
     */
    public Cnaps(int features, int hidden, int task, boolean adaptive, RandomGenerator random) {
        this.hidden = hidden;
        this.adaptive = adaptive;
        this.encW = weight(features, task, 1, random);
        this.encB = Var.param(Tensor.zeros(Shape.of(task)));
        this.w1 = weight(features, hidden, 2, random);
        this.b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
        this.w2 = weight(hidden, hidden, 2, random);
        this.b2 = Var.param(Tensor.zeros(Shape.of(hidden)));
        // FiLM generators start near zero, so the extractor starts unmodulated
        this.film1 = Var.param(Tensor.normal(Shape.of(task, 2 * hidden), 0f, 0.01f, random));
        this.film2 = Var.param(Tensor.normal(Shape.of(task, 2 * hidden), 0f, 0.01f, random));
        this.headW = Var.param(Tensor.normal(Shape.of(hidden, hidden + 1), 0f, 0.01f, random));
        this.headB = Var.param(Tensor.zeros(Shape.of(hidden + 1)));
    }

    private static Var weight(int in, int out, double gain, RandomGenerator random) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0f, (float) Math.sqrt(gain / in), random));
    }

    /**
     * The task vector of a support set.
     *
     * @param support {@code [n, features]}
     * @return {@code [1, task]}
     */
    public Var taskEmbedding(Tensor support) {
        return Var.of(support).matmul(encW).add(encB).relu().mean(0, true);
    }

    /**
     * The task-adapted features.
     *
     * @param x {@code [n, features]}
     * @param task the task vector {@code [1, task]}
     * @return {@code [n, hidden]}
     */
    public Var features(Tensor x, Var task) {
        int n = x.shape().dim(0);
        Var h = Var.of(x).matmul(w1).add(b1);
        h = modulate(h, task, film1, n).relu();
        h = h.matmul(w2).add(b2);
        return modulate(h, task, film2, n).relu();
    }

    private Var modulate(Var h, Var task, Var generator, int n) {
        if (!adaptive) {
            return h;
        }
        Var gb = task.matmul(generator);
        Shape rows = Shape.of(n, hidden);
        return Functional.film(h, gb.slice(1, 0, hidden).expand(rows), gb.slice(1, hidden, 2 * hidden).expand(rows));
    }

    /**
     * The query logits of an episode, from one forward pass over its support set.
     *
     * @param episode the episode
     * @return {@code [queries, ways]}
     */
    public Var logits(FewShotEpisode episode) {
        Var task = adaptive ? taskEmbedding(episode.support()) : null;
        Var means = PrototypicalNetworks.prototypes(features(episode.support(), task), episode.supportLabels(),
                episode.ways());
        Var head = means.matmul(headW).add(headB);
        Var weights = head.slice(1, 0, hidden).add(means);
        Var bias = head.slice(1, hidden, hidden + 1).transpose();
        return features(episode.query(), task).matmul(weights.transpose()).add(bias);
    }

    /**
     * The episodic training loss.
     *
     * @param episode the episode
     * @return the query cross-entropy
     */
    public Var loss(FewShotEpisode episode) {
        return Functional.crossEntropy(logits(episode), episode.queryIds());
    }

    /**
     * Classifies an episode's queries.
     *
     * @param episode the episode
     * @return the predicted class of each query
     */
    public int[] classify(FewShotEpisode episode) {
        return FewShotEpisode.argmax(logits(episode).value());
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>(List.of(w1, b1, w2, b2, headW, headB));
        if (adaptive) {
            out.addAll(List.of(encW, encB, film1, film2));
        }
        return out;
    }
}
