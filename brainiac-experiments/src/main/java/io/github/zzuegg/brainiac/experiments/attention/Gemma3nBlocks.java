package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.AltUpBlock;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.NestedFeedForward;
import io.github.zzuegg.brainiac.experimental.nn.PerLayerEmbeddingBlock;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.ResidualMixerBlock;
import io.github.zzuegg.brainiac.experimental.nn.WeightPruning;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The Gemma 3n family of efficiency blocks on the {@link ArchitectureBench} toy language.
 *
 * <ul>
 *   <li><b>MatFormer</b> ({@link NestedFeedForward}): one stack trained so that every FFN's
 *       first {@code m} hidden units work alone; its loss at each extracted width against a
 *       model trained at that width from scratch, and against extracting the same widths
 *       from a normally trained model, by truncation or by magnitude pruning
 *       ({@link WeightPruning#magnitude}) to the same number of FFN weights.</li>
 *   <li><b>Per-layer embeddings</b> ({@link PerLayerEmbeddingBlock}) and <b>AltUp</b>
 *       ({@link AltUpBlock}): loss against the "hot" parameters that cost compute each step,
 *       compared with a plain stack of the same block width (equal compute) and a plain stack
 *       twice as wide.</li>
 * </ul>
 */
public final class Gemma3nBlocks {

    private static final int DEPTH = 2;

    private Gemma3nBlocks() {
    }

    /**
     * One extracted width's losses.
     *
     * @param hidden the FFN hidden width
     * @param nested the MatFormer model at this width
     * @param scratch a model trained at this width from scratch (NaN when not trained)
     * @param truncated the normally trained full model truncated to this width
     * @param pruned the normally trained full model magnitude-pruned to as many FFN weights
     */
    public record WidthRow(int hidden, double nested, double scratch, double truncated, double pruned) {
    }

    /**
     * One efficiency configuration.
     *
     * @param name the configuration
     * @param validationLoss the validation loss
     * @param hotParameters parameters used in matrix products every step (body only)
     * @param coldParameters lookup-only parameters (per-layer tables)
     */
    public record EfficiencyRow(String name, double validationLoss, long hotParameters, long coldParameters) {
    }

    private static LayerSpec attention() {
        return RefinedAttention.of(2).causal(true).rotary(10000f);
    }

    private static LayerSpec ffnStack(int hidden, int[] granularities) {
        return LayerChain.repeat(ResidualMixerBlock.of(attention()).ffn(NestedFeedForward.spec(hidden,
                granularities)), DEPTH);
    }

    /**
     * Trains the nested model and the baselines and measures every width.
     *
     * @param width the model width
     * @param hidden the full FFN hidden width (a multiple of 8)
     * @param steps training steps per model
     * @param seed the seed
     * @param scratchWidths the widths to also train from scratch
     * @return one row per granularity, smallest first
     */
    public static List<WidthRow> matformer(int width, int hidden, int steps, long seed, int[] scratchWidths) {
        int[] widths = NestedFeedForward.halvings(hidden);
        ArchitectureBench.LmResult nested = ArchitectureBench.languageModel("MatFormer", ffnStack(hidden, widths),
                width, steps, seed);
        ArchitectureBench.LmResult plain = ArchitectureBench.languageModel("plain", ffnStack(hidden,
                new int[] {hidden}), width, steps, seed);
        List<WidthRow> rows = new ArrayList<>();
        for (int m : widths) {
            double scratch = Double.NaN;
            for (int s : scratchWidths) {
                if (s == m) {
                    scratch = ArchitectureBench.languageModel("scratch " + m, ffnStack(m, new int[] {m}), width, steps,
                            seed).validationLoss();
                }
            }
            rows.add(new WidthRow(m, lossAt(nested, m), scratch, lossAt(plain, m), prunedLoss(plain, hidden, m)));
        }
        setWidth(plain.body(), hidden);
        return rows;
    }

    /** The validation loss with every nested FFN set to {@code m} hidden units. */
    private static double lossAt(ArchitectureBench.LmResult r, int m) {
        setWidth(r.body(), m);
        double loss = ArchitectureBench.nextTokenLoss(r.model(), r.validation(), false).value().item();
        setWidth(r.body(), nestedHidden(r.body()));
        return loss;
    }

    /** Magnitude-prunes every FFN of a full model to the weights of an {@code m}-unit FFN, measures, restores. */
    private static double prunedLoss(ArchitectureBench.LmResult r, int hidden, int m) {
        List<NestedFeedForward> ffns = ffns(r.body());
        List<Tensor> saved = new ArrayList<>();
        for (NestedFeedForward f : ffns) {
            for (Var p : f.parameters()) {
                saved.add(p.value());
                p.assign(WeightPruning.magnitude(p.value(), 1 - m / (double) hidden));
            }
        }
        double loss = ArchitectureBench.nextTokenLoss(r.model(), r.validation(), false).value().item();
        int i = 0;
        for (NestedFeedForward f : ffns) {
            for (Var p : f.parameters()) {
                p.assign(saved.get(i++));
            }
        }
        return loss;
    }

    private static int nestedHidden(Layer body) {
        int[] g = ffns(body).getFirst().granularities();
        return g[g.length - 1];
    }

    private static void setWidth(Layer body, int m) {
        for (NestedFeedForward f : ffns(body)) {
            f.activeWidth(m);
        }
    }

    /**
     * Finds the nested FFNs of a body.
     *
     * @param body the body
     * @return the FFNs in order
     */
    public static List<NestedFeedForward> ffns(Layer body) {
        List<NestedFeedForward> out = new ArrayList<>();
        if (body instanceof LayerChain c) {
            for (Layer l : c.layers()) {
                if (l instanceof ResidualMixerBlock b && b.ffn() instanceof NestedFeedForward f) {
                    out.add(f);
                }
            }
        }
        return out;
    }

    /**
     * Plain stacks at width {@code d} and {@code 2d}, AltUp with two slices of width {@code d},
     * and per-layer embeddings of width {@code d / 2} on a width-{@code d} stack.
     *
     * @param width the block width {@code d}
     * @param steps training steps
     * @param seed the seed
     * @return plain d, plain 2d, AltUp, PLE
     */
    public static List<EfficiencyRow> efficiency(int width, int steps, long seed) {
        List<EfficiencyRow> rows = new ArrayList<>();
        LayerSpec block = ResidualMixerBlock.of(attention());
        ArchitectureBench.LmResult plain = ArchitectureBench.languageModel("plain", LayerChain.repeat(block, DEPTH),
                width, steps, seed);
        rows.add(new EfficiencyRow("plain d=" + width, plain.validationLoss(), plain.parameters(), 0));
        ArchitectureBench.LmResult wide = ArchitectureBench.languageModel("wide", LayerChain.repeat(block, DEPTH),
                2 * width, steps, seed);
        rows.add(new EfficiencyRow("plain d=" + 2 * width, wide.validationLoss(), wide.parameters(), 0));
        ArchitectureBench.LmResult altUp = ArchitectureBench.languageModel("AltUp", AltUpBlock.stack(block, 2, DEPTH),
                2 * width, steps, seed);
        rows.add(new EfficiencyRow("AltUp K=2, d=" + width, altUp.validationLoss(), altUp.parameters(), 0));
        rows.add(perLayerEmbeddings(block, width, width / 2, steps, seed));
        return rows;
    }

    /**
     * Trains a stack of per-layer-embedding blocks with the same recipe as
     * {@link ArchitectureBench#languageModel} (token plus previous-token embedding, Adam 1e-2,
     * batches of 16).
     */
    private static EfficiencyRow perLayerEmbeddings(LayerSpec block, int width, int perLayer, int steps, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int vocab = ArchitectureBench.VOCAB;
        int length = ArchitectureBench.LENGTH;
        ArchitectureBench.TinyModel m = ArchitectureBench.model(LayerChain.repeat(block, 0), vocab, length, width,
                true, random);
        List<PerLayerEmbeddingBlock> blocks = new ArrayList<>();
        for (int l = 0; l < DEPTH; l++) {
            blocks.add(new PerLayerEmbeddingBlock(block.create(Shape.of(length, width), random), width, vocab,
                    perLayer, random));
        }
        List<Var> params = new ArrayList<>(m.parameters());
        blocks.forEach(b -> params.addAll(b.parameters()));
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            int[][] batch = ArchitectureBench.corpus(random, 16);
            loss(m, blocks, batch, true).backward();
            opt.step(params);
        }
        int[][] validation = ArchitectureBench.corpus(new SplittableRandom(seed + 999), 128);
        double loss = loss(m, blocks, validation, false).value().item();
        long hot = 0;
        long cold = 0;
        for (PerLayerEmbeddingBlock b : blocks) {
            hot += count(b.hotParameters());
            cold += count(b.coldParameters());
        }
        return new EfficiencyRow("PLE p=" + perLayer + ", d=" + width, loss, hot, cold);
    }

    private static Var loss(ArchitectureBench.TinyModel m, List<PerLayerEmbeddingBlock> blocks, int[][] tokens,
            boolean training) {
        int b = tokens.length;
        int t = tokens[0].length;
        float[] ids = new float[b * t];
        float[] next = new float[b * (t - 1)];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                ids[i * t + j] = tokens[i][j];
                if (j < t - 1) {
                    next[i * (t - 1) + j] = tokens[i][j + 1];
                }
            }
        }
        Tensor idTensor = Tensor.of(Shape.of(b, t), ids);
        Var h = m.hidden(tokens, training);
        for (PerLayerEmbeddingBlock blk : blocks) {
            h = blk.forward(h, idTensor, training);
        }
        int vocab = m.readout().shape().dim(1);
        Var logits = m.readout(h).slice(1, 0, t - 1).reshape(b * (t - 1), vocab);
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(next.length), next));
    }

    private static long count(List<Var> vars) {
        long n = 0;
        for (Var v : vars) {
            n += v.value().size();
        }
        return n;
    }

    /**
     * Formats width rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String widthTable(List<WidthRow> rows) {
        StringBuilder sb = new StringBuilder(String.format("%7s %9s %9s %10s %9s%n", "hidden", "MatFormer", "scratch",
                "truncated", "pruned"));
        for (WidthRow r : rows) {
            sb.append(String.format("%7d %9.4f %9.4f %10.4f %9.4f%n", r.hidden(), r.nested(), r.scratch(),
                    r.truncated(), r.pruned()));
        }
        return sb.toString();
    }

    /**
     * Formats efficiency rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String efficiencyTable(List<EfficiencyRow> rows) {
        StringBuilder sb = new StringBuilder(String.format("%-20s %9s %8s %8s%n", "model", "val loss", "hot", "cold"));
        for (EfficiencyRow r : rows) {
            sb.append(String.format("%-20s %9.4f %8d %8d%n", r.name(), r.validationLoss(), r.hotParameters(),
                    r.coldParameters()));
        }
        return sb.toString();
    }

    /**
     * Runs both comparisons.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.print(widthTable(matformer(16, 64, 400, 1, new int[] {8, 16, 32, 64})));
        System.out.print(efficiencyTable(efficiency(16, 400, 1)));
    }
}
