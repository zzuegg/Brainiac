package io.github.zzuegg.brainiac.experimental.ood;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A classifier split into a feature extractor ({@code body}, any {@link Model}) and a linear
 * head, so post-hoc OOD scores, activation shaping, last-layer retraining and adaptation
 * methods can reach the penultimate features. {@link #model()} joins both into one
 * sequential {@link Model} that shares their layers, so core tools (test-time adaptation,
 * checkpoints, {@code Trainer}) see one model.
 */
public final class FeatureClassifier {

    private final Model body;
    private final Dense head;
    private final Model joined;

    /**
     * Joins a body and a head.
     *
     * @param body the feature extractor; its output is {@code [n, features]}
     * @param head the linear head {@code features -> classes}
     */
    public FeatureClassifier(Model body, Dense head) {
        this.body = Objects.requireNonNull(body);
        this.head = Objects.requireNonNull(head);
        Model.Builder b = Model.sequential().input(body.inputShape().toArray());
        for (Layer l : body.layers()) {
            b.layer(l);
        }
        this.joined = b.layer(head).build();
    }

    /**
     * An MLP with ReLU hidden layers.
     *
     * @param inputs the input features
     * @param hidden the hidden widths; the last is the feature size
     * @param classes the number of classes
     * @param seed the seed
     * @return the classifier
     */
    public static FeatureClassifier mlp(int inputs, int[] hidden, int classes, long seed) {
        Model.Builder b = Model.sequential().input(inputs);
        for (int h : hidden) {
            b.dense(h, Activation.RELU);
        }
        Model body = b.seed(seed).build();
        Dense head = new Dense(hidden[hidden.length - 1], classes, Activation.IDENTITY, Init.XAVIER,
                new SplittableRandom(seed + 991));
        return new FeatureClassifier(body, head);
    }

    /** @return the feature extractor */
    public Model body() {
        return body;
    }

    /** @return the linear head */
    public Dense head() {
        return head;
    }

    /** @return body and head as one model sharing their layers */
    public Model model() {
        return joined;
    }

    /** @return the number of classes */
    public int classes() {
        return head.outputSize(0);
    }

    /**
     * @param x the inputs
     * @param training training mode (dropout, batch statistics)
     * @return the penultimate features
     */
    public Var features(Var x, boolean training) {
        return body.forward(x, training);
    }

    /**
     * @param x the inputs
     * @param training training mode
     * @return the logits
     */
    public Var logits(Var x, boolean training) {
        return head.forward(features(x, training), training);
    }

    /**
     * @param x the inputs
     * @return the features in inference mode, without a graph
     */
    public Tensor features(Tensor x) {
        return Var.noGrad(() -> body.forward(Var.of(x), false).value());
    }

    /**
     * @param x the inputs
     * @return the logits in inference mode, without a graph
     */
    public Tensor logits(Tensor x) {
        return Var.noGrad(() -> logits(Var.of(x), false).value());
    }

    /** @return all parameters, body first */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(body.parameters());
        p.addAll(head.parameters());
        return p;
    }

    /** @return copies of every parameter value, in {@link #parameters()} order */
    public List<Tensor> snapshot() {
        List<Tensor> out = new ArrayList<>();
        for (Var v : parameters()) {
            out.add(v.value().copy());
        }
        for (Var v : body.buffers()) {
            out.add(v.value().copy());
        }
        return out;
    }

    /**
     * Restores parameter (and buffer) values taken by {@link #snapshot()}.
     *
     * @param values the snapshot
     */
    public void restore(List<Tensor> values) {
        List<Var> all = new ArrayList<>(parameters());
        all.addAll(body.buffers());
        for (int i = 0; i < all.size(); i++) {
            all.get(i).assign(values.get(i).copy());
        }
    }

    /**
     * Predicted classes.
     *
     * @param x the inputs
     * @return the argmax of the logits
     */
    public int[] predict(Tensor x) {
        return logits(x).argmax(1).toIntArray();
    }

    /**
     * Accuracy.
     *
     * @param x the inputs
     * @param y the labels
     * @return the share predicted right
     */
    public double accuracy(Tensor x, int[] y) {
        int[] p = predict(x);
        int c = 0;
        for (int i = 0; i < y.length; i++) {
            c += p[i] == y[i] ? 1 : 0;
        }
        return c / (double) y.length;
    }
}
