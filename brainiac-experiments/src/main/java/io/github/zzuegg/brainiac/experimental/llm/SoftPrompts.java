package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Prompt-based parameter-efficient fine-tuning of a frozen {@link HookedDecoder}: only a few
 * learned vectors in front of the input change.
 * <ul>
 * <li>{@link #promptTuning}: prompt tuning (Lester et al. 2021,
 *     <a href="https://arxiv.org/abs/2104.08691">arXiv:2104.08691</a>), learned input
 *     embeddings read before the tokens, started from the embeddings of real tokens (the
 *     start the paper found best).</li>
 * <li>{@link #pTuning}: P-tuning (Liu et al. 2021, <a href="https://arxiv.org/abs/2103.10385">arXiv:2103.10385</a>),
 *     the same virtual tokens but produced by a small MLP prompt encoder from free seeds,
 *     which couples them and makes optimization steadier.</li>
 * <li>{@link #prefixTuning}: prefix tuning (Li and Liang 2021,
 *     <a href="https://arxiv.org/abs/2101.00190">arXiv:2101.00190</a>) and P-tuning v2: learned
 *     keys and values prepended at <em>every</em> attention layer, started from the keys and
 *     values the model computes for real tokens.</li>
 * </ul>
 * Each returns the trainable values; train only those.
 */
public final class SoftPrompts {

    private SoftPrompts() {
    }

    /**
     * @param model the frozen decoder
     * @param length the virtual tokens
     * @param seed picks the real tokens the prompt starts from
     * @return the trainable prompt
     */
    public static List<Var> promptTuning(HookedDecoder model, int length, long seed) {
        Var prompt = Var.param(embeddingsOf(model, length, seed));
        model.softPrompt(prompt);
        return List.of(prompt);
    }

    /**
     * @param model the frozen decoder
     * @param length the virtual tokens
     * @param hidden the encoder's hidden width
     * @param seed the seed
     * @return the encoder's trainable values
     */
    public static List<Var> pTuning(HookedDecoder model, int length, int hidden, long seed) {
        int dim = model.settings().dim();
        SplittableRandom r = new SplittableRandom(seed);
        Var seeds = Var.param(embeddingsOf(model, length, seed));
        Var w1 = Var.param(Tensor.normal(Shape.of(dim, hidden), 0, (float) (1 / Math.sqrt(dim)), r));
        Var b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
        Var w2 = Var.param(Tensor.zeros(Shape.of(hidden, dim)));
        Var b2 = Var.param(Tensor.zeros(Shape.of(dim)));
        List<Var> params = List.of(seeds, w1, b1, w2, b2);
        model.softPrompt(new HookedDecoder.PromptEncoder() {
            @Override
            public Var prompt() {
                Var h = seeds.matmul(w1).add(b1).activation(ActivationOp.GELU, 0);
                return seeds.add(h.matmul(w2).add(b2));
            }

            @Override
            public List<Var> parameters() {
                return params;
            }
        });
        return params;
    }

    /**
     * @param model the frozen decoder
     * @param length the prefix length per layer
     * @param seed picks the real tokens whose keys and values start the prefixes
     * @return the trainable keys and values of every layer
     */
    public static List<Var> prefixTuning(HookedDecoder model, int length, long seed) {
        Tensor e = embeddingsOf(model, length, seed);
        List<Var> params = new ArrayList<>();
        for (int i = 0; i < model.layers(); i++) {
            Layer k = model.linears().get("block." + i + ".attn.k");
            Layer v = model.linears().get("block." + i + ".attn.v");
            Var pk = Var.param(k.forward(Var.of(e)).value());
            Var pv = Var.param(v.forward(Var.of(e)).value());
            model.prefix(i, pk, pv);
            params.add(pk);
            params.add(pv);
        }
        return params;
    }

    /**
     * Removes every soft prompt and prefix.
     *
     * @param model the decoder
     */
    public static void clear(HookedDecoder model) {
        model.softPrompt((Var) null);
        for (int i = 0; i < model.layers(); i++) {
            model.prefix(i, null, null);
        }
    }

    private static Tensor embeddingsOf(HookedDecoder model, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] ids = new float[length];
        for (int i = 0; i < length; i++) {
            ids[i] = r.nextInt(model.vocabSize());
        }
        return model.embedding().value().gather(0, Tensor.of(Shape.of(length), ids));
    }
}
