package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GluFeedForward;
import io.github.zzuegg.brainiac.experimental.nn.PositionalAttention.Position;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock.Norm;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock.Placement;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock.Residual;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Compares normalization and residual schemes in a deep, narrow transformer (8 blocks of
 * width 16) trained without warm-up at a high learning rate, the regime where they matter:
 * post-LN, pre-LN, peri-LN (sandwich), pre-LN with QK-norm, Dynamic Tanh, ReZero without any
 * normalization, LayerScale and DeepNorm. The task is a sequence regression that needs
 * attention: each output step is the running mean of the inputs so far, scaled by a fixed
 * random matrix.
 */
public final class NormalizationShootout {

    private NormalizationShootout() {
    }

    /** @return the compared block descriptions, by name */
    public static Map<String, TunableTransformerBlock.Spec> variants() {
        TunableTransformerBlock.Spec base = TunableTransformerBlock.of(2).causal(true).position(Position.ROTARY)
                .ffn(GluFeedForward.Kind.SWIGLU).ffnFactor(2).depth(8);
        Map<String, TunableTransformerBlock.Spec> m = new LinkedHashMap<>();
        m.put("post-LN", base.placement(Placement.POST));
        m.put("pre-LN", base);
        m.put("peri-LN", base.placement(Placement.PERI));
        m.put("pre-LN + QK-norm", base.qkNorm(true));
        m.put("DyT", base.norm(Norm.DYT));
        m.put("ReZero", base.norm(Norm.NONE).residual(Residual.REZERO));
        m.put("LayerScale", base.residual(Residual.LAYER_SCALE));
        m.put("DeepNorm", base.residual(Residual.DEEP_NORM));
        return m;
    }

    /**
     * Trains one variant.
     *
     * @param block the block description
     * @param steps training steps
     * @param learningRate the Adam learning rate (constant, no warm-up)
     * @param seed the seed
     * @return {@code {first loss, mean loss over the last 10 steps}}
     */
    public static float[] run(TunableTransformerBlock.Spec block, int steps, float learningRate, long seed) {
        int t = 8;
        int d = 16;
        Model.Builder b = Model.sequential().input(t, d).seed(seed);
        for (int i = 0; i < 8; i++) {
            b.add(block);
        }
        Model model = b.build();
        SplittableRandom rnd = new SplittableRandom(seed);
        Tensor mix = Tensor.normal(Shape.of(d, d), 0, (float) (1 / Math.sqrt(d)), rnd);
        float[] tri = new float[t * t];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j <= i; j++) {
                tri[i * t + j] = 1f / (i + 1);
            }
        }
        Tensor runningMean = Tensor.of(Shape.of(t, t), tri);
        Optimizer opt = Optimizer.adam(learningRate);
        float first = 0;
        double tail = 0;
        for (int s = 0; s < steps; s++) {
            Tensor x = Tensor.normal(Shape.of(16, t, d), 0, 1, rnd);
            Tensor y = runningMean.matmul(x).matmul(mix);
            Var loss = model.forward(Var.of(x), true).sub(y).square().mean();
            float v = loss.value().item();
            first = s == 0 ? v : first;
            tail += s >= steps - 10 ? (Float.isFinite(v) ? v : 1e9) / 10 : 0;
            loss.backward();
            opt.step(model.parameters());
        }
        return new float[] {first, (float) tail};
    }
}
