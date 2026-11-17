package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.experimental.nn.DifferentialAttention;
import io.github.zzuegg.brainiac.experimental.nn.FeedbackTransformer;
import io.github.zzuegg.brainiac.experimental.nn.ForgettingAttention;
import io.github.zzuegg.brainiac.experimental.nn.GatedAttention;
import io.github.zzuegg.brainiac.experimental.nn.HybridLinearStack;
import io.github.zzuegg.brainiac.experimental.nn.LatentAttention;
import io.github.zzuegg.brainiac.experimental.nn.NativeSparseAttention;
import io.github.zzuegg.brainiac.experimental.nn.NormalizedTransformerBlock;
import io.github.zzuegg.brainiac.experimental.nn.ParallelHybridBlock;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.SelectiveAttention;
import io.github.zzuegg.brainiac.experimental.nn.SigmoidAttention;
import io.github.zzuegg.brainiac.experimental.nn.Softpick;
import io.github.zzuegg.brainiac.experimental.nn.StickBreakingAttention;
import io.github.zzuegg.brainiac.experimental.nn.TalkingHeadsAttention;
import io.github.zzuegg.brainiac.experimental.nn.UniversalTransformer;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.TransformerBlock;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Attention variants side by side on the {@link MixerShootout} diagnostic tasks: the core
 * {@link TransformerBlock} as the baseline against differential, gated (NoPE and rotary),
 * forgetting, natively sparse and normalized (nGPT) attention, the Universal and Feedback
 * transformers, a Kimi-style 3:1 linear/full hybrid, and the refinements of
 * {@link KvEfficientAttention}, {@link AttentionWeightings}, {@link LengthGeneralisation} and
 * {@link SelectiveContext} (latent, sigmoid, Softpick, stick-breaking, talking-heads,
 * selective and sink-logit attention, and a Hymba-style parallel hybrid). Recall is where softmax attention
 * shines; recent recall shows which variants carry order information without explicit
 * positions (forget gates, rotary, recency bias); parity shows the depth and recurrence
 * advantage of looped and feedback models.
 *
 * <p>Papers: Ye et al., "Differential Transformer",
 * <a href="https://arxiv.org/abs/2410.05258">arXiv:2410.05258</a>; Qiu et al., "Gated
 * Attention for Large Language Models", <a href="https://arxiv.org/abs/2505.06708">arXiv:2505.06708</a>;
 * Lin et al., "Forgetting Transformer", <a href="https://arxiv.org/abs/2503.02130">arXiv:2503.02130</a>;
 * Yuan et al., "Native Sparse Attention", <a href="https://arxiv.org/abs/2502.11089">arXiv:2502.11089</a>;
 * Loshchilov et al., "nGPT", <a href="https://arxiv.org/abs/2410.01131">arXiv:2410.01131</a>.
 */
public final class AttentionShootout {

    private AttentionShootout() {
    }

    /**
     * The attention variants of this theme, causal, with two heads.
     *
     * @return name to spec, in a stable order, the core transformer block first
     */
    public static Map<String, LayerSpec> lineup() {
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("transformer", TransformerBlock.of(2).causal(true).rmsNorm(true));
        m.put("differential", DifferentialAttention.of(2, true, 1));
        m.put("gated NoPE", GatedAttention.of(2, true, 0f));
        m.put("gated rotary", GatedAttention.of(2, true, 10000f));
        m.put("forgetting", ForgettingAttention.of(2, true));
        m.put("native sparse", NativeSparseAttention.of(2, 2, 2, 2));
        m.put("nGPT", NormalizedTransformerBlock.of(2, 32, true, 0.1f));
        m.put("universal", UniversalTransformer.of(2, true, 3));
        m.put("feedback", FeedbackTransformer.of(2, 2));
        m.put("hybrid 3:1", HybridLinearStack.of(2, 4, 3));
        m.put("latent (MLA)", LatentAttention.of(2, 8, 4));
        m.put("sigmoid", SigmoidAttention.of(2, true, 10000f));
        m.put("softpick", Softpick.of(2, true, 10000f));
        m.put("stick-breaking", StickBreakingAttention.of(2));
        m.put("talking heads", TalkingHeadsAttention.of(2, true, 10000f));
        m.put("selective", SelectiveAttention.of(2, 10000f));
        m.put("sink logit", RefinedAttention.of(2).causal(true).rotary(10000f).sinkLogit(true));
        m.put("parallel hybrid", ParallelHybridBlock.of(2, 1, 2));
        return m;
    }

    /**
     * Runs the lineup on recall, recent recall and parity and prints the table.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        List<MixerShootout.Task> tasks = List.of(MixerShootout.recall(8, 8, 4), MixerShootout.recentRecall(3, 4, 4),
                MixerShootout.parity(5));
        System.out.print(MixerShootout.table(MixerShootout.run(lineup(), tasks, 16, 300, 0.01f, 1)));
    }
}
