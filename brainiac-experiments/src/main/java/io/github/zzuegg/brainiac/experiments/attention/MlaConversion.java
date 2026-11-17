package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.MlaConverter;
import io.github.zzuegg.brainiac.experimental.nn.LatentAttention;
import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * GQA-to-MLA conversion ({@link MlaConverter}) on the {@link ArchitectureBench} toy language:
 * a two-block model with grouped-query attention (4 query heads, one KV head, RoPE) is
 * trained, converted to {@link LatentAttention} at two latent sizes (one as large as the GQA
 * cache, one compressing it), and fine-tuned for a few steps; compared with the original and
 * with MLA of the same sizes trained from scratch for the full budget. Reported: validation
 * loss against cached values per token (all layers).
 */
public final class MlaConversion {

    /**
     * One configuration.
     *
     * @param name the configuration
     * @param cachedPerToken cached values per token over all layers
     * @param validationLoss the validation loss
     */
    public record Row(String name, int cachedPerToken, double validationLoss) {
    }

    private static final int WIDTH = 32;
    private static final int HEADS = 4;
    private static final int DEPTH = 2;
    private static final int ROPE = 4;

    private MlaConversion() {
    }

    /**
     * Runs the comparison.
     *
     * @param steps pretraining steps (and the from-scratch MLA budget)
     * @param fineTune fine-tuning steps after conversion
     * @param latents the latent sizes to convert to
     * @param seed the seed
     * @return GQA, GQA with the fine-tuning steps, then per latent: converted, converted and fine-tuned,
     *     from scratch
     */
    public static List<Row> run(int steps, int fineTune, int[] latents, long seed) {
        MultiHeadAttention.Spec gqa = MultiHeadAttention.of(HEADS).kvHeads(1).causal(true).rotary(10000f).bias(false);
        ArchitectureBench.LmResult base = ArchitectureBench.languageModel("GQA", ArchitectureBench.stack(gqa, DEPTH),
                WIDTH, steps, seed);
        ArchitectureBench.TinyModel original = base.model();
        int[][] validation = base.validation();
        List<Row> rows = new ArrayList<>();
        rows.add(new Row("GQA kv=1", DEPTH * 2 * (WIDTH / HEADS), base.validationLoss()));
        List<Var> params = original.parameters();
        List<Tensor> saved = new ArrayList<>();
        for (Var p : params) {
            saved.add(p.value());
        }
        int[][] calibration = ArchitectureBench.corpus(new SplittableRandom(seed + 5), 32);
        for (int latent : latents) {
            int cache = DEPTH * (latent + ROPE);
            Tensor input = embedded(original, calibration);
            LayerChain body = MlaConverter.convert((LayerChain) original.body(), latent, ROPE, input);
            ArchitectureBench.TinyModel converted = new ArchitectureBench.TinyModel(original.embed(),
                    original.previous(), body, original.readout());
            rows.add(new Row("converted latent=" + latent, cache, loss(converted, validation)));
            fineTune(converted, fineTune, seed);
            rows.add(new Row("converted+" + fineTune + " latent=" + latent, cache, loss(converted, validation)));
            for (int i = 0; i < params.size(); i++) {   // the converted model shares embeddings and FFNs
                params.get(i).assign(saved.get(i));
            }
            ArchitectureBench.LmResult scratch = ArchitectureBench.languageModel("MLA", ArchitectureBench.stack(
                    LatentAttention.of(HEADS, latent, ROPE), DEPTH), WIDTH, steps, seed);
            rows.add(new Row("scratch MLA latent=" + latent, cache, scratch.validationLoss()));
        }
        fineTune(original, fineTune, seed);   // the same extra steps for the original, for fairness
        rows.add(1, new Row("GQA kv=1 +" + fineTune, DEPTH * 2 * (WIDTH / HEADS), loss(original, validation)));
        return rows;
    }

    private static void fineTune(ArchitectureBench.TinyModel m, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 11);
        Optimizer opt = Optimizer.adam(3e-3f);
        List<Var> trainable = m.parameters();
        for (int s = 0; s < steps; s++) {
            ArchitectureBench.nextTokenLoss(m, ArchitectureBench.corpus(r, 16), true).backward();
            opt.step(trainable);
        }
    }

    private static double loss(ArchitectureBench.TinyModel m, int[][] tokens) {
        return ArchitectureBench.nextTokenLoss(m, tokens, false).value().item();
    }

    /** The body's input: token plus previous-token embeddings. */
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
        Var x = m.embed().gather(0, Tensor.of(Shape.of(b * t), ids));
        if (m.previous() != null) {
            x = x.add(m.previous().gather(0, Tensor.of(Shape.of(b * t), prev)));
        }
        return x.value().reshape(b, t, WIDTH);
    }
}
