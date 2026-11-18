package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Concept bottleneck models (Koh et al., ICML 2020, <a href="https://arxiv.org/abs/2007.04612">arXiv:2007.04612</a>)
 * and concept embedding models (Espinosa Zarlenga et al., NeurIPS 2022,
 * <a href="https://arxiv.org/abs/2209.09056">arXiv:2209.09056</a>): an encoder predicts
 * {@code k} human concepts from the input and a small head predicts the label only from them, so
 * a user can read and correct ({@link #intervene}) the concepts at test time.
 *
 * <ul>
 *   <li>{@link Mode#INDEPENDENT}: encoder and head trained separately, the head on the true
 *       concepts, used on the predicted probabilities.</li>
 *   <li>{@link Mode#SEQUENTIAL}: the head trained on the frozen encoder's probabilities.</li>
 *   <li>{@link Mode#JOINT_SOFT}: both trained together on task cross-entropy plus
 *       {@code lambda} times the concept loss; soft probabilities can leak extra label
 *       information.</li>
 *   <li>{@link Mode#JOINT_HARD}: joint, but the head sees thresholded concepts (straight-through
 *       gradient), which blocks leakage.</li>
 *   <li>{@link Mode#CEM}: each concept has a positive and a negative embedding computed from the
 *       input, mixed by the concept probability from a shared scoring function; the head reads
 *       the mixed embeddings. Intervening sets the mixing probability to the true concept.</li>
 * </ul>
 * As a {@link Layer} the forward pass returns the task logits.
 */
public final class ConceptBottleneck implements Layer {

    /** How the bottleneck is trained. */
    public enum Mode {
        /** Independent encoder and head. */
        INDEPENDENT,
        /** Head on the frozen encoder. */
        SEQUENTIAL,
        /** Joint, soft probabilities. */
        JOINT_SOFT,
        /** Joint, thresholded concepts. */
        JOINT_HARD,
        /** Concept embedding model. */
        CEM
    }

    private static final int EMBED = 8;

    private final Mode mode;
    private final int concepts;
    private final int classes;
    private final Var w1;
    private final Var b1;
    private final Var wc;
    private final Var bc;
    private final Var wPos;
    private final Var wNeg;
    private final Var score;
    private final Var h1;
    private final Var hb1;
    private final Var h2;
    private final Var hb2;

    /**
     * Creates an untrained model.
     *
     * @param mode the training mode
     * @param inputs input features
     * @param hidden the encoder's hidden width
     * @param concepts the number of binary concepts
     * @param classes the number of task classes
     * @param seed the initial weights
     */
    public ConceptBottleneck(Mode mode, int inputs, int hidden, int concepts, int classes, long seed) {
        this.mode = mode;
        this.concepts = concepts;
        this.classes = classes;
        SplittableRandom r = new SplittableRandom(seed);
        w1 = Var.param(Tensor.normal(Shape.of(inputs, hidden), 0, (float) Math.sqrt(2.0 / inputs), r));
        b1 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        wc = Var.param(Tensor.normal(Shape.of(hidden, concepts), 0, (float) Math.sqrt(1.0 / hidden), r));
        bc = Var.param(Tensor.zeros(Shape.of(1, concepts)));
        wPos = Var.param(Tensor.normal(Shape.of(hidden, concepts * EMBED), 0, (float) Math.sqrt(1.0 / hidden), r));
        wNeg = Var.param(Tensor.normal(Shape.of(hidden, concepts * EMBED), 0, (float) Math.sqrt(1.0 / hidden), r));
        score = Var.param(Tensor.normal(Shape.of(2 * EMBED, 1), 0, (float) Math.sqrt(1.0 / EMBED), r));
        int headIn = mode == Mode.CEM ? concepts * EMBED : concepts;
        h1 = Var.param(Tensor.normal(Shape.of(headIn, 16), 0, (float) Math.sqrt(2.0 / headIn), r));
        hb1 = Var.param(Tensor.zeros(Shape.of(1, 16)));
        h2 = Var.param(Tensor.normal(Shape.of(16, classes), 0, 0.25f, r));
        hb2 = Var.param(Tensor.zeros(Shape.of(1, classes)));
    }

    /** @return the mode */
    public Mode mode() {
        return mode;
    }

    private Var hidden(Var x) {
        return x.matmul(w1).add(b1).relu();
    }

    /** The concept logits (or, for CEM, the scores of the mixed embeddings) and embeddings. */
    private Var[] encode(Var x) {
        Var h = hidden(x);
        if (mode != Mode.CEM) {
            return new Var[] {h.matmul(wc).add(bc), null, null};
        }
        int b = x.shape().dim(0);
        Var pos = h.matmul(wPos).leakyRelu(0.1f).reshape(b, concepts, EMBED);
        Var neg = h.matmul(wNeg).leakyRelu(0.1f).reshape(b, concepts, EMBED);
        Var both = Var.concat(2, pos, neg).reshape(b * concepts, 2 * EMBED);
        Var logits = both.matmul(score).reshape(b, concepts);
        return new Var[] {logits, pos, neg};
    }

    /**
     * The head's input from concept probabilities.
     *
     * @param probs concept probabilities {@code [b, k]}
     * @param enc the encoding (embeddings for CEM)
     * @return the head input
     */
    private Var bottleneck(Var probs, Var[] enc) {
        if (mode == Mode.CEM) {
            int b = probs.shape().dim(0);
            Var p = probs.reshape(b, concepts, 1);
            Var mixed = enc[1].mul(p).add(enc[2].mul(p.neg().add(1f)));
            return mixed.reshape(b, concepts * EMBED);
        }
        if (mode == Mode.JOINT_HARD) {
            Tensor hard = probs.greater(0.5f);
            return probs.add(Var.of(hard.sub(probs.value())));
        }
        return probs;
    }

    private Var head(Var in) {
        return in.matmul(h1).add(hb1).relu().matmul(h2).add(hb2);
    }

    @Override
    public Var forward(Var input) {
        return logits(input, null, null);
    }

    /**
     * Task logits with some concepts replaced by their true values.
     *
     * @param input the inputs
     * @param intervene the concepts to correct, or null
     * @param truth the true concepts {@code [b, k]} (0/1), or null
     * @return {@code [b, classes]}
     */
    public Var logits(Var input, int[] intervene, Tensor truth) {
        Var[] enc = encode(input);
        Var probs = enc[0].sigmoid();
        if (intervene != null && intervene.length > 0) {
            int b = input.shape().dim(0);
            float[] keep = new float[b * concepts];
            java.util.Arrays.fill(keep, 1f);
            for (int i = 0; i < b; i++) {
                for (int c : intervene) {
                    keep[i * concepts + c] = 0f;
                }
            }
            Tensor k = Tensor.of(Shape.of(b, concepts), keep);
            probs = probs.mul(k).add(Var.of(truth.mul(k.neg().add(1f))));
            if (mode == Mode.JOINT_HARD) {
                probs = Var.of(probs.value());
            }
        }
        return head(bottleneck(probs, enc));
    }

    /**
     * The predicted concept probabilities.
     *
     * @param input the inputs
     * @return {@code [b, k]}
     */
    public Tensor concepts(Tensor input) {
        return Var.noGrad(() -> encode(Var.of(input))[0].sigmoid()).value();
    }

    /**
     * Predicted classes after an intervention.
     *
     * @param input the inputs
     * @param intervene the concepts to correct (empty for none)
     * @param truth the true concepts
     * @return the classes
     */
    public int[] intervene(Tensor input, int[] intervene, Tensor truth) {
        Tensor l = Var.noGrad(() -> logits(Var.of(input), intervene, truth)).value();
        float[] a = l.argmax(1).toArray();
        int[] out = new int[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = (int) a[i];
        }
        return out;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(encoderParameters());
        p.addAll(headParameters());
        return p;
    }

    private List<Var> encoderParameters() {
        return mode == Mode.CEM ? List.of(w1, b1, wPos, wNeg, score) : List.of(w1, b1, wc, bc);
    }

    private List<Var> headParameters() {
        return List.of(h1, hb1, h2, hb2);
    }

    @Override
    public int outputSize(int inputSize) {
        return classes;
    }

    private static Var conceptLoss(Var logits, Tensor c) {
        // binary cross-entropy with logits: softplus(z) - c z
        return logits.softplus().sub(logits.mul(c)).mean();
    }

    /**
     * Trains the model in its mode.
     *
     * @param x the inputs {@code [n, d]}
     * @param c the true concepts {@code [n, k]} (0/1)
     * @param y the labels {@code [n]}
     * @param steps steps per stage
     * @param lambda the concept-loss weight of the joint modes
     * @param seed the batch seed
     */
    public void fit(Tensor x, Tensor c, Tensor y, int steps, float lambda, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.shape().dim(0);
        int batch = Math.min(64, n);
        switch (mode) {
            case INDEPENDENT, SEQUENTIAL -> {
                Optimizer enc = Optimizer.adam(3e-3f);
                for (int s = 0; s < steps; s++) {
                    int from = r.nextInt(n - batch + 1);
                    conceptLoss(encode(Var.of(x.slice(0, from, from + batch)))[0], c.slice(0, from, from + batch))
                            .backward();
                    enc.step(encoderParameters());
                }
                Optimizer head = Optimizer.adam(3e-3f);
                Tensor input = mode == Mode.INDEPENDENT ? c : concepts(x);
                for (int s = 0; s < steps; s++) {
                    int from = r.nextInt(n - batch + 1);
                    Functional.crossEntropy(head(Var.of(input.slice(0, from, from + batch))),
                            y.slice(0, from, from + batch)).backward();
                    head.step(headParameters());
                    encoderParameters().forEach(Var::zeroGrad);
                }
            }
            default -> {
                Optimizer opt = Optimizer.adam(3e-3f);
                for (int s = 0; s < steps; s++) {
                    int from = r.nextInt(n - batch + 1);
                    Var xb = Var.of(x.slice(0, from, from + batch));
                    Var[] enc = encode(xb);
                    Var task = Functional.crossEntropy(head(bottleneck(enc[0].sigmoid(), enc)),
                            y.slice(0, from, from + batch));
                    task.add(conceptLoss(enc[0], c.slice(0, from, from + batch)).mul(lambda)).backward();
                    opt.step(parameters());
                }
            }
        }
    }
}
