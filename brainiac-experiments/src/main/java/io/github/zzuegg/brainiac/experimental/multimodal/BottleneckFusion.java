package io.github.zzuegg.brainiac.experimental.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Attention bottlenecks for multimodal fusion (MBT, Nagrani et al., NeurIPS 2021) with
 * modality dropout (ModDrop, Neverova et al., 2016). Each modality has its own transformer
 * layers (core {@link TransformerBlock}s); below the fusion layer the streams are separate,
 * and from it upwards cross-modal information flows only through a few shared bottleneck
 * tokens: each stream attends over its own tokens plus the bottleneck, and the new
 * bottleneck is the average of the streams' updated copies. This forces each modality to
 * condense what it shares, at a fraction of the cost of full attention over all tokens.
 * Each stream's mean-pooled tokens feed its own classifier and the logits are averaged.
 *
 * <p>Modality dropout zeroes a whole modality's input during training with a given
 * probability (never all of them), so the fused model degrades gracefully when one is
 * missing at test time, where "missing" means the same zero input.
 *
 * <p>{@link Mode#EARLY} concatenates all tokens into one stream with full attention;
 * {@link Mode#LATE} never fuses (a fusion layer above the top), so it averages unimodal
 * classifiers. With one modality it is a plain unimodal transformer classifier.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2107.00135">arXiv:2107.00135</a>,
 * <a href="https://arxiv.org/abs/1501.00102">arXiv:1501.00102</a>.
 */
public final class BottleneckFusion {

    /** How the modalities meet. */
    public enum Mode {
        /** Separate streams that exchange information through bottleneck tokens. */
        BOTTLENECK,
        /** One stream over all modalities' tokens. */
        EARLY,
        /** Separate streams, averaged logits only. */
        LATE
    }

    private final int[] tokens;
    private final int dim;
    private final int layers;
    private final int fusionLayer;
    private final int classes;
    private final Mode mode;
    private final List<Var> inProject = new ArrayList<>();
    private final List<Var> positions = new ArrayList<>();
    private final List<List<Layer>> streams = new ArrayList<>();
    private final Var bottleneck;
    private final List<Var> heads = new ArrayList<>();
    private final List<Var> headBias = new ArrayList<>();

    /**
     * Creates the model.
     *
     * @param tokens per modality its token count
     * @param features per modality its token width
     * @param dim the model width
     * @param layers blocks per stream
     * @param fusionLayer the first layer that uses the bottleneck (ignored unless BOTTLENECK)
     * @param bottlenecks the bottleneck tokens
     * @param classes output classes
     * @param mode the fusion mode
     * @param seed the seed
     */
    public BottleneckFusion(int[] tokens, int[] features, int dim, int layers, int fusionLayer, int bottlenecks,
            int classes, Mode mode, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.tokens = tokens.clone();
        this.dim = dim;
        this.layers = layers;
        this.fusionLayer = mode == Mode.BOTTLENECK ? fusionLayer : layers;
        this.classes = classes;
        this.mode = mode;
        int m = tokens.length;
        for (int i = 0; i < m; i++) {
            inProject.add(Var.param(Tensor.normal(Shape.of(features[i], dim), 0, (float) Math.sqrt(1.0 / features[i]),
                    r)));
            positions.add(Var.param(Tensor.normal(Shape.of(1, tokens[i], dim), 0, 0.1f, r)));
        }
        int nStreams = mode == Mode.EARLY ? 1 : m;
        for (int s = 0; s < nStreams; s++) {
            List<Layer> blocks = new ArrayList<>();
            for (int l = 0; l < layers; l++) {
                blocks.add(TransformerBlock.of(4).ffnSize(2 * dim).create(Shape.of(8, dim), r));
            }
            streams.add(blocks);
            heads.add(Var.param(Tensor.normal(Shape.of(dim, classes), 0, (float) Math.sqrt(1.0 / dim), r)));
            headBias.add(Var.param(Tensor.zeros(Shape.of(classes))));
        }
        this.bottleneck = Var.param(Tensor.normal(Shape.of(1, Math.max(1, bottlenecks), dim), 0, 0.1f, r));
    }

    /**
     * Logits for a batch.
     *
     * @param inputs per modality {@code [batch, tokens, features]}
     * @param present per modality whether it is present (absent ones are zeroed)
     * @param training whether training
     * @return {@code [batch, classes]}
     */
    public Var logits(List<Tensor> inputs, boolean[] present, boolean training) {
        int m = inputs.size();
        int b = inputs.get(0).shape().dim(0);
        List<Var> x = new ArrayList<>();
        for (int i = 0; i < m; i++) {
            Tensor in = present[i] ? inputs.get(i) : Tensor.zeros(inputs.get(i).shape());
            x.add(Var.of(in).matmul(inProject.get(i)).add(positions.get(i)));
        }
        if (mode == Mode.EARLY) {
            Var h = Var.concat(1, x.toArray(new Var[0]));
            for (Layer l : streams.get(0)) {
                h = l.forward(h, training);
            }
            return h.mean(1, false).matmul(heads.get(0)).add(headBias.get(0));
        }
        int nb = bottleneck.shape().dim(1);
        Var z = bottleneck.expand(Shape.of(b, nb, dim));
        for (int l = 0; l < layers; l++) {
            if (l < fusionLayer) {
                for (int i = 0; i < m; i++) {
                    x.set(i, streams.get(i).get(l).forward(x.get(i), training));
                }
            } else {
                Var sum = null;
                for (int i = 0; i < m; i++) {
                    Var h = streams.get(i).get(l).forward(Var.concat(1, x.get(i), z), training);
                    x.set(i, h.slice(1, 0, tokens[i]));
                    Var zi = h.slice(1, tokens[i], tokens[i] + nb);
                    sum = sum == null ? zi : sum.add(zi);
                }
                z = sum.mul(1f / m);
            }
        }
        Var logits = null;
        for (int i = 0; i < m; i++) {
            Var li = x.get(i).mean(1, false).matmul(heads.get(i)).add(headBias.get(i));
            logits = logits == null ? li : logits.add(li);
        }
        return logits.mul(1f / m);
    }

    /**
     * The training loss with modality dropout.
     *
     * @param inputs per modality {@code [batch, tokens, features]}
     * @param labels class ids {@code [batch]}
     * @param dropout the probability of dropping each modality (at least one is kept)
     * @param random the dropout source
     * @return the cross-entropy
     */
    public Var loss(List<Tensor> inputs, Tensor labels, double dropout, SplittableRandom random) {
        int m = inputs.size();
        boolean[] present = new boolean[m];
        boolean any = false;
        for (int i = 0; i < m; i++) {
            present[i] = random.nextDouble() >= dropout;
            any |= present[i];
        }
        if (!any) {
            present[random.nextInt(m)] = true;
        }
        return Functional.crossEntropy(logits(inputs, present, true), labels);
    }

    /**
     * Accuracy with some modalities missing.
     *
     * @param inputs per modality inputs
     * @param labels class ids
     * @param present which modalities are present
     * @return the share classified correctly
     */
    public double accuracy(List<Tensor> inputs, Tensor labels, boolean[] present) {
        float[] pred = Var.noGrad(() -> logits(inputs, present, false)).value().argmax(1).toArray();
        float[] y = labels.toArray();
        int hit = 0;
        for (int i = 0; i < y.length; i++) {
            hit += pred[i] == y[i] ? 1 : 0;
        }
        return (double) hit / y.length;
    }

    private long blockMacs(int n) {
        return 4L * n * dim * dim + 2L * n * n * dim + 4L * n * dim * dim;
    }

    /** @return multiply-adds per example of the transformer blocks */
    public long macs() {
        int nb = bottleneck.shape().dim(1);
        if (mode == Mode.EARLY) {
            int n = 0;
            for (int t : tokens) {
                n += t;
            }
            return layers * blockMacs(n);
        }
        long total = 0;
        for (int l = 0; l < layers; l++) {
            for (int t : tokens) {
                total += blockMacs(l < fusionLayer ? t : t + nb);
            }
        }
        return total;
    }

    /** @return every trainable value */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(inProject);
        p.addAll(positions);
        for (List<Layer> s : streams) {
            for (Layer l : s) {
                p.addAll(l.parameters());
            }
        }
        if (mode == Mode.BOTTLENECK) {
            p.add(bottleneck);
        }
        p.addAll(heads);
        p.addAll(headBias);
        return p;
    }

    @Override
    public String toString() {
        return "BottleneckFusion(" + mode + ", fusion at " + fusionLayer + ", " + classes + " classes)";
    }
}
