package io.github.zzuegg.brainiac.experiments.compression;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.compress.FixedSparsityOptimizer;
import io.github.zzuegg.brainiac.experimental.compress.StructuredLmPruner;
import io.github.zzuegg.brainiac.experimental.kernels.BlockSparseMatmul;
import io.github.zzuegg.brainiac.experimental.kernels.NmSparseMatmul;
import io.github.zzuegg.brainiac.experimental.nn.GluFeedForward;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.ResidualMixerBlock;
import io.github.zzuegg.brainiac.experimental.nn.SliceCompressor;
import io.github.zzuegg.brainiac.experimental.nn.WeightPruning;
import io.github.zzuegg.brainiac.experiments.attention.ArchitectureBench;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Structured pruning of the toy language model of {@code language.SliceCompression} (wave 8 item
 * 8.5.5): a pre-norm transformer of {@value #DEPTH} blocks of width {@value #WIDTH} (four RoPE
 * heads, SwiGLU of width 64) trained on the {@link ArchitectureBench} toy language. At each share
 * of block-matrix weights removed it compares LLM-Pruner, FASP and 2SSP (via
 * {@link StructuredLmPruner}) with SliceGPT, unstructured Wanda and dropping whole blocks, by
 * validation loss straight after pruning and after a short recovery fine-tune (masked, so removed
 * structures stay removed). Used by {@link PruningShootout#languageModel}.
 */
final class LmStructuredPruning {

    static final int WIDTH = 32;
    static final int DEPTH = 2;

    private LmStructuredPruning() {
    }

    /** Calibration activations of one block. */
    private record Capture(Tensor a, Tensor z, Tensor c, Tensor hidden) {
    }

    static List<PruningShootout.Row> run(double[] shares, int steps, int recovery, long seed) {
        MultiHeadAttention.Spec mha = MultiHeadAttention.of(4).causal(true).rotary(10000f).bias(false);
        ArchitectureBench.LmResult base = ArchitectureBench.languageModel("MHA", ArchitectureBench.stack(mha, DEPTH),
                WIDTH, steps, seed);
        ArchitectureBench.TinyModel m = base.model();
        int[][] val = base.validation();
        LayerChain body = (LayerChain) m.body();
        int[][] calib = ArchitectureBench.corpus(new SplittableRandom(seed + 3), 32);
        List<Var> params = m.parameters();
        List<Tensor> saved = values(params);
        List<Var> matrices = new ArrayList<>();
        for (Var p : body.parameters()) {
            if (p.shape().rank() == 2) {
                matrices.add(p);
            }
        }
        long total = 0;
        for (Var w : matrices) {
            total += w.shape().size();
        }
        List<PruningShootout.Row> rows = new ArrayList<>();
        rows.add(new PruningShootout.Row("dense", 0, loss(m, val), loss(m, val)));
        for (double share : shares) {
            for (String method : List.of("llm-pruner", "fasp", "2ssp", "wanda")) {
                restore(params, saved);
                List<Capture> caps = capture(m, body, calib);
                long removed = switch (method) {
                    case "llm-pruner" -> taylor(m, body, calib, share);
                    case "fasp" -> fasp(body, caps, share);
                    case "2ssp" -> twoStage(m, body, caps, calib, share, total);
                    default -> wanda(body, caps, share);
                };
                double before = loss(m, val);
                Optimizer opt = FixedSparsityOptimizer.ofZeros(Optimizer.adam(3e-3f), matrices);
                fineTune(m, opt, recovery, seed);
                rows.add(new PruningShootout.Row(method, removed / (double) total, loss(m, val), before));
            }
            restore(params, saved);
            int keep = (int) Math.round(WIDTH * (1 - share));
            SliceCompressor.Sliced s = SliceCompressor.slice(List.of(m.embed(), m.previous()), body, m.readout(),
                    embedded(m, calib), keep);
            ArchitectureBench.TinyModel sliced = new ArchitectureBench.TinyModel(Var.param(s.embeddings().get(0)),
                    Var.param(s.embeddings().get(1)), s.body(), Var.param(s.readout()));
            double before = loss(sliced, val);
            fineTune(sliced, Optimizer.adam(3e-3f), recovery, seed);
            rows.add(new PruningShootout.Row("slicegpt", 1 - keep / (double) WIDTH, loss(sliced, val), before));
            int drop = (int) Math.floor(DEPTH * share + 1e-9);
            if (drop >= 1 && drop < DEPTH) {
                restore(params, saved);
                List<Layer> kept = new ArrayList<>(body.layers().subList(0, DEPTH - drop));   // drop the last blocks
                ArchitectureBench.TinyModel dropped = new ArchitectureBench.TinyModel(m.embed(), m.previous(),
                        new LayerChain(kept), m.readout());
                double b = loss(dropped, val);
                fineTune(dropped, Optimizer.adam(3e-3f), recovery, seed);
                rows.add(new PruningShootout.Row("drop " + drop + " block", drop / (double) DEPTH, loss(dropped, val),
                        b));
            }
        }
        restore(params, saved);
        return rows;
    }

    /**
     * The same language model pruned to 50% in the kernel-friendly formats of item 8.5.18:
     * unstructured magnitude, 2:4 along the input dimension, and 8x8 and 4x4 blocks by block
     * norm, each followed by a masked recovery fine-tune.
     */
    static List<PruningShootout.Row> sparseFormats(int steps, int recovery, long seed) {
        MultiHeadAttention.Spec mha = MultiHeadAttention.of(4).causal(true).rotary(10000f).bias(false);
        ArchitectureBench.LmResult base = ArchitectureBench.languageModel("MHA", ArchitectureBench.stack(mha, DEPTH),
                WIDTH, steps, seed);
        ArchitectureBench.TinyModel m = base.model();
        int[][] val = base.validation();
        List<Var> params = m.parameters();
        List<Tensor> saved = values(params);
        List<Var> matrices = new ArrayList<>();
        for (Var p : m.body().parameters()) {
            if (p.shape().rank() == 2) {
                matrices.add(p);
            }
        }
        List<PruningShootout.Row> rows = new ArrayList<>();
        rows.add(new PruningShootout.Row("dense", 0, loss(m, val), loss(m, val)));
        for (String method : List.of("unstructured", "2:4", "block 4x4", "block 8x8")) {
            restore(params, saved);
            long zeros = 0;
            long total = 0;
            for (Var w : matrices) {
                Tensor pruned = switch (method) {
                    case "unstructured" -> WeightPruning.magnitude(w.value(), 0.5);
                    case "2:4" -> NmSparseMatmul.prune(w.value(), 2, 4);
                    case "block 4x4" -> BlockSparseMatmul.prune(w.value(), 4, 0.5);
                    default -> BlockSparseMatmul.prune(w.value(), 8, 0.5);
                };
                w.assign(pruned);
                for (float v : pruned.toArray()) {
                    zeros += v == 0 ? 1 : 0;
                }
                total += pruned.shape().size();
            }
            double before = loss(m, val);
            fineTune(m, FixedSparsityOptimizer.ofZeros(Optimizer.adam(3e-3f), matrices), recovery, seed);
            rows.add(new PruningShootout.Row(method, zeros / (double) total, loss(m, val), before));
        }
        restore(params, saved);
        return rows;
    }

    private static long taylor(ArchitectureBench.TinyModel m, LayerChain body, int[][] calib, double share) {
        List<Var> w = new ArrayList<>();
        for (Var p : body.parameters()) {
            if (p.shape().rank() == 2) {
                w.add(p);
            }
        }
        List<Var> g = Var.gradients(ArchitectureBench.nextTokenLoss(m, calib, false), w, false);
        Map<Var, Tensor> grads = new IdentityHashMap<>();
        for (int i = 0; i < w.size(); i++) {
            grads.put(w.get(i), g.get(i).value());
        }
        long removed = 0;
        int i = 0;
        for (Layer l : body.layers()) {
            ResidualMixerBlock b = (ResidualMixerBlock) l;
            for (List<StructuredLmPruner.Group> groups : List.of(
                    StructuredLmPruner.heads((MultiHeadAttention) b.mixer(), "b" + i + "."),
                    StructuredLmPruner.neurons((GluFeedForward) b.ffn(), "b" + i + "."))) {
                boolean[] rm = StructuredLmPruner.lowestShare(StructuredLmPruner.taylorScores(groups, grads), share);
                StructuredLmPruner.remove(groups, rm);
                removed += StructuredLmPruner.params(groups, rm);
            }
            i++;
        }
        return removed;
    }

    private static long fasp(LayerChain body, List<Capture> caps, double share) {
        long removed = 0;
        for (int i = 0; i < body.layers().size(); i++) {
            ResidualMixerBlock b = (ResidualMixerBlock) body.layers().get(i);
            MultiHeadAttention att = (MultiHeadAttention) b.mixer();
            GluFeedForward ffn = (GluFeedForward) b.ffn();
            Var wo = att.parameters().get(3);
            Var down = ffn.parameters().get(1);
            Map<Var, Tensor> inputs = new IdentityHashMap<>();
            inputs.put(wo, caps.get(i).z());
            inputs.put(down, caps.get(i).hidden());
            for (List<StructuredLmPruner.Group> groups : List.of(StructuredLmPruner.heads(att, "h"),
                    StructuredLmPruner.neurons(ffn, "n"))) {
                Var consumer = groups.getFirst().consumer().weight();
                Tensor original = consumer.value();
                boolean[] rm = StructuredLmPruner.lowestShare(StructuredLmPruner.faspScores(groups, inputs), share);
                StructuredLmPruner.remove(groups, rm);
                removed += StructuredLmPruner.params(groups, rm);
                StructuredLmPruner.restoreConsumer(consumer, original, inputs.get(consumer),
                        StructuredLmPruner.keptRows(groups, rm, consumer.shape().dim(0)), 1e-3);
            }
        }
        return removed;
    }

    private static long twoStage(ArchitectureBench.TinyModel m, LayerChain body, List<Capture> caps, int[][] calib,
            double share, long total) {
        long budget = Math.round(share * total);
        int attnParams = 4 * WIDTH * WIDTH;
        int modules = (int) Math.min(DEPTH, budget / 2 / attnParams);   // half the budget at most goes to depth
        long removed = 0;
        // stage 1: width, by mean output activation of each neuron, same share per block
        long neuronBudget = budget - (long) modules * attnParams;
        int perNeuron = 3 * WIDTH;
        int hidden = ((GluFeedForward) ((ResidualMixerBlock) body.layers().getFirst()).ffn()).parameters().get(0)
                .shape().dim(1);
        double neuronShare = Math.min(1, neuronBudget / (double) (perNeuron * hidden * DEPTH));
        for (int i = 0; i < DEPTH; i++) {
            ResidualMixerBlock b = (ResidualMixerBlock) body.layers().get(i);
            GluFeedForward ffn = (GluFeedForward) b.ffn();
            List<StructuredLmPruner.Group> groups = StructuredLmPruner.neurons(ffn, "n");
            Map<Var, Tensor> inputs = new IdentityHashMap<>();
            inputs.put(ffn.parameters().get(1), caps.get(i).hidden());
            boolean[] rm = StructuredLmPruner.lowestShare(StructuredLmPruner.activationScores(groups, inputs),
                    neuronShare);
            StructuredLmPruner.remove(groups, rm);
            removed += StructuredLmPruner.params(groups, rm);
        }
        // stage 2: depth, removing whole attention modules one at a time by calibration loss
        boolean[] gone = new boolean[DEPTH];
        for (int k = 0; k < modules; k++) {
            int best = -1;
            double bestLoss = Double.MAX_VALUE;
            for (int i = 0; i < DEPTH; i++) {
                if (gone[i]) {
                    continue;
                }
                MultiHeadAttention att = (MultiHeadAttention) ((ResidualMixerBlock) body.layers().get(i)).mixer();
                List<Tensor> keep = values(att.parameters());
                StructuredLmPruner.Group g = StructuredLmPruner.module(att, "a");
                StructuredLmPruner.remove(List.of(g), new boolean[] {true});
                double l = loss(m, calib);
                restore(att.parameters(), keep);
                if (l < bestLoss) {
                    bestLoss = l;
                    best = i;
                }
            }
            MultiHeadAttention att = (MultiHeadAttention) ((ResidualMixerBlock) body.layers().get(best)).mixer();
            StructuredLmPruner.Group g = StructuredLmPruner.module(att, "a");
            StructuredLmPruner.remove(List.of(g), new boolean[] {true});
            removed += g.params();
            gone[best] = true;
        }
        return removed;
    }

    private static long wanda(LayerChain body, List<Capture> caps, double share) {
        long removed = 0;
        for (int i = 0; i < body.layers().size(); i++) {
            ResidualMixerBlock b = (ResidualMixerBlock) body.layers().get(i);
            List<Var> w = b.mixer().parameters();
            Capture c = caps.get(i);
            for (int j = 0; j < 3; j++) {
                w.get(j).assign(WeightPruning.wanda(w.get(j).value(), c.a(), share));
            }
            w.get(3).assign(WeightPruning.wanda(w.get(3).value(), c.z(), share));
            List<Var> f = b.ffn().parameters();   // up, down, gate
            f.get(0).assign(WeightPruning.wanda(f.get(0).value(), c.c(), share));
            f.get(2).assign(WeightPruning.wanda(f.get(2).value(), c.c(), share));
            f.get(1).assign(WeightPruning.wanda(f.get(1).value(), c.hidden(), share));
            for (Var v : List.of(w.get(0), w.get(1), w.get(2), w.get(3), f.get(0), f.get(1), f.get(2))) {
                removed += Math.round(WeightPruning.sparsity(v.value()) * v.shape().size());
            }
        }
        return removed;
    }

    /** Replays every block on calibration tokens and records the inputs of each projection. */
    private static List<Capture> capture(ArchitectureBench.TinyModel m, LayerChain body, int[][] tokens) {
        Tensor input = embedded(m, tokens);
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        List<Capture> caps = new ArrayList<>();
        Var x = Var.of(input);
        for (Layer l : body.layers()) {
            ResidualMixerBlock block = (ResidualMixerBlock) l;
            MultiHeadAttention att = (MultiHeadAttention) block.mixer();
            GluFeedForward ffn = (GluFeedForward) block.ffn();
            List<Var> gains = block.parameters();
            Var a = rms(x).mul(gains.get(0));
            Tensor z = heads(a, att.parameters(), att.settings(), b, t);
            Var h = x.add(att.forward(a));
            Var c = rms(h).mul(gains.get(1));
            List<Var> f = ffn.parameters();
            Var hid = ffn.kind().activation().forward(c.matmul(f.get(2))).mul(c.matmul(f.get(0)));
            int hw = hid.shape().dim(2);
            caps.add(new Capture(a.value().reshape(b * t, WIDTH), z, c.value().reshape(b * t, WIDTH),
                    hid.value().reshape(b * t, hw)));
            x = block.forward(x);
        }
        return caps;
    }

    /** The concatenated head outputs, the input of the output projection. */
    private static Tensor heads(Var a, List<Var> w, MultiHeadAttention.Settings s, int b, int t) {
        Var q = a.matmul(w.get(0)).reshape(b, t, s.heads(), s.headSize()).permute(0, 2, 1, 3);
        Var k = a.matmul(w.get(1)).reshape(b, t, s.kvHeads(), s.headSize()).permute(0, 2, 1, 3);
        Var v = a.matmul(w.get(2)).reshape(b, t, s.kvHeads(), s.valueSize()).permute(0, 2, 1, 3);
        q = Functional.rotary(q, s.rotaryBase(), 0);
        k = Functional.rotary(k, s.rotaryBase(), 0);
        float scale = s.scale() != 0 ? s.scale() : (float) (1 / Math.sqrt(s.headSize()));
        Var o = Functional.attention(q, k, v, true, scale);
        return o.permute(0, 2, 1, 3).reshape(b * t, s.heads() * s.valueSize()).value();
    }

    private static Var rms(Var x) {
        return x.div(x.square().mean(-1, true).add(1e-6f).sqrt());
    }

    private static Tensor embedded(ArchitectureBench.TinyModel m, int[][] tokens) {
        int b = tokens.length;
        int t = tokens[0].length;
        float[] ids = new float[b * t];
        float[] prev = new float[b * t];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                ids[i * t + j] = tokens[i][j];
                prev[i * t + j] = tokens[i][Math.max(0, j - 1)];
            }
        }
        Var x = m.embed().gather(0, Tensor.of(Shape.of(b * t), ids))
                .add(m.previous().gather(0, Tensor.of(Shape.of(b * t), prev)));
        return x.value().reshape(b, t, WIDTH);
    }

    private static void fineTune(ArchitectureBench.TinyModel m, Optimizer opt, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 21);
        List<Var> params = m.parameters();
        for (int s = 0; s < steps; s++) {
            ArchitectureBench.nextTokenLoss(m, ArchitectureBench.corpus(r, 16), true).backward();
            opt.step(params);
        }
    }

    static double loss(ArchitectureBench.TinyModel m, int[][] tokens) {
        return Var.noGrad(() -> ArchitectureBench.nextTokenLoss(m, tokens, false).value().item());
    }

    private static List<Tensor> values(List<Var> params) {
        List<Tensor> v = new ArrayList<>();
        for (Var p : params) {
            v.add(p.value());
        }
        return v;
    }

    private static void restore(List<Var> params, List<Tensor> values) {
        for (int i = 0; i < params.size(); i++) {
            params.get(i).assign(values.get(i));
        }
    }
}
