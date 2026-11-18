package io.github.zzuegg.brainiac.experiments.sequence;

import io.github.zzuegg.brainiac.experimental.nn.GatedDeltaNet;
import io.github.zzuegg.brainiac.experimental.nn.KimiDeltaAttention;
import io.github.zzuegg.brainiac.experimental.nn.LargeChunkTttLayer;
import io.github.zzuegg.brainiac.experimental.nn.MesaLayer;
import io.github.zzuegg.brainiac.experimental.nn.TestTimeTrainingLayer;
import io.github.zzuegg.brainiac.experimental.nn.TitansLayer;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The 2025 "test-time regression" mixers on the {@link MixerShootout} associative-recall
 * diagnostic, against the delta-rule family: {@link MesaLayer} (MesaNet, the in-context
 * least-squares problem solved exactly per token by conjugate gradient,
 * <a href="https://arxiv.org/abs/2506.05233">arXiv:2506.05233</a>) and
 * {@link LargeChunkTttLayer} (LaCT, one fast-weight update per large chunk plus local attention,
 * <a href="https://arxiv.org/abs/2505.23884">arXiv:2505.23884</a>), compared with
 * {@link GatedDeltaNet} and Kimi Delta Attention (one delta-rule step per token),
 * {@link TestTimeTrainingLayer} (TTT-Linear, a gradient step per mini-batch of four) and
 * {@link TitansLayer} (a surprise-driven neural memory). Run through
 * {@link DeltaFamilyShootout#run}.
 */
public final class TestTimeRegressionShootout {

    private TestTimeRegressionShootout() {
    }

    /**
     * The lineup.
     *
     * @return name to mixer spec
     */
    public static Map<String, LayerSpec> lineup() {
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("MesaNet", MesaLayer.of(2, 4));
        m.put("LaCT", LargeChunkTttLayer.of(2, 4, true));
        m.put("LaCT (no local attention)", LargeChunkTttLayer.of(2, 4, false));
        m.put("Gated DeltaNet", GatedDeltaNet.of(2, GatedDeltaNet.Gating.SCALAR, 4));
        m.put("KDA", KimiDeltaAttention.of(2, 4, 4));
        m.put("TTT-Linear", TestTimeTrainingLayer.of(2, TestTimeTrainingLayer.Variant.LINEAR, 4, 0.5f));
        m.put("Titans", TitansLayer.spec(2, 8));
        return m;
    }
}
