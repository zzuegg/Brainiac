package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * The learning-rate range test (Smith 2017, <a href="https://arxiv.org/abs/1506.01186">arXiv:1506.01186</a>;
 * popularised as fast.ai's LR finder): train for a few hundred steps while the learning
 * rate grows exponentially from tiny to huge, record the smoothed loss, and pick a rate
 * where the loss falls fastest, about ten times below the rate where it is lowest. The
 * weights are restored afterwards; use a fresh optimizer for the real run.
 */
public final class LrRangeTest {

    private LrRangeTest() {
    }

    /**
     * The recorded curve and the suggestion.
     *
     * @param learningRates the rate at every step
     * @param losses the smoothed loss at every step
     * @param suggested the suggested learning rate
     */
    public record Result(float[] learningRates, float[] losses, float suggested) {

        /** @return the rate where the smoothed loss was lowest */
        public float minimumLossRate() {
            int best = 0;
            for (int i = 1; i < losses.length; i++) {
                if (losses[i] < losses[best]) {
                    best = i;
                }
            }
            return learningRates[best];
        }
    }

    /**
     * Runs the test.
     *
     * @param parameters the parameters to train
     * @param optimizer a fresh optimizer whose learning rate the test sets
     * @param loss computes the loss of the next batch (not yet differentiated)
     * @param minRate the first learning rate, for example {@code 1e-6f}
     * @param maxRate the last, for example 10
     * @param steps the number of steps, for example 100
     * @return the curve and the suggested rate
     */
    public static Result run(List<Var> parameters, Optimizer optimizer, Supplier<Var> loss, float minRate,
            float maxRate, int steps) {
        List<Tensor> saved = new ArrayList<>();
        for (Var p : parameters) {
            saved.add(p.value());
        }
        List<Float> rates = new ArrayList<>();
        List<Float> losses = new ArrayList<>();
        double smooth = 0;
        double best = Double.POSITIVE_INFINITY;
        double ratio = Math.pow(maxRate / minRate, 1.0 / Math.max(1, steps - 1));
        for (int i = 0; i < steps; i++) {
            float lr = (float) (minRate * Math.pow(ratio, i));
            optimizer.learningRate(lr);
            Var l = loss.get();
            l.backward();
            float value = l.value().item();
            optimizer.step(parameters);
            smooth = 0.9 * smooth + 0.1 * value;
            double corrected = smooth / (1 - Math.pow(0.9, i + 1));   // bias-corrected average
            rates.add(lr);
            losses.add((float) corrected);
            best = Math.min(best, corrected);
            if (!Double.isFinite(corrected) || corrected > 4 * best && i > 10) {
                break;
            }
        }
        for (int i = 0; i < parameters.size(); i++) {
            parameters.get(i).assign(saved.get(i));
            parameters.get(i).zeroGrad();
        }
        float[] r = new float[rates.size()];
        float[] ls = new float[losses.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = rates.get(i);
            ls[i] = losses.get(i);
        }
        // steepest descent of the loss per log-rate, before the minimum
        int minIdx = 0;
        for (int i = 1; i < ls.length; i++) {
            if (ls[i] < ls[minIdx]) {
                minIdx = i;
            }
        }
        int steepest = 0;
        double slope = 0;
        for (int i = 1; i <= minIdx; i++) {
            double s = ls[i] - ls[i - 1];
            if (s < slope) {
                slope = s;
                steepest = i;
            }
        }
        float suggestion = minIdx == 0 ? r[0] : Math.min(r[steepest], r[minIdx] / 10 * 3);
        return new Result(r, ls, suggestion);
    }
}
