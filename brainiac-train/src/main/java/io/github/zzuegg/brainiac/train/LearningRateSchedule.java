package io.github.zzuegg.brainiac.train;

/**
 * Chooses the learning rate for each epoch. Starting high and lowering the rate later
 * often trains faster and ends closer to the best weights.
 *
 * <pre>{@code
 * Trainer.of(model)
 *         .optimizer(Optimizer.adam(0.01f))
 *         .schedule(LearningRateSchedule.cosine(200, 0.0001f))
 *         .epochs(200)
 *         .fit(data);
 * }</pre>
 */
@FunctionalInterface
public interface LearningRateSchedule {

    /**
     * Returns the learning rate for an epoch.
     *
     * @param epoch the epoch, starting at 1
     * @param baseRate the optimizer's initial learning rate
     * @return the rate to use, positive
     */
    float rate(int epoch, float baseRate);

    /**
     * Multiplies the rate by {@code factor} every {@code everyEpochs} epochs.
     *
     * @param factor for example {@code 0.5f} to halve
     * @param everyEpochs how often to apply it
     * @return the schedule
     */
    static LearningRateSchedule step(float factor, int everyEpochs) {
        if (everyEpochs < 1) {
            throw new IllegalArgumentException("everyEpochs must be at least 1, got " + everyEpochs);
        }
        return (epoch, base) -> base * (float) Math.pow(factor, (epoch - 1) / everyEpochs);
    }

    /**
     * Multiplies the rate by {@code decay} every epoch.
     *
     * @param decay for example {@code 0.99f}
     * @return the schedule
     */
    static LearningRateSchedule exponential(float decay) {
        return (epoch, base) -> base * (float) Math.pow(decay, epoch - 1);
    }

    /**
     * Lowers the rate along a half cosine from the base rate to {@code minRate} over
     * {@code totalEpochs}, then stays at {@code minRate}.
     *
     * @param totalEpochs the length of the curve
     * @param minRate the final rate
     * @return the schedule
     */
    static LearningRateSchedule cosine(int totalEpochs, float minRate) {
        if (totalEpochs < 1) {
            throw new IllegalArgumentException("totalEpochs must be at least 1, got " + totalEpochs);
        }
        return (epoch, base) -> {
            double t = Math.min(1.0, (epoch - 1) / (double) Math.max(1, totalEpochs - 1));
            return (float) (minRate + (base - minRate) * 0.5 * (1 + Math.cos(Math.PI * t)));
        };
    }

    /**
     * Raises the rate linearly from a small value to the base over the first epochs, then
     * hands over to another schedule. Helps unstable starts.
     *
     * @param epochs the warm-up length
     * @param then the schedule after warm-up, counted from the first epoch
     * @return the schedule
     */
    static LearningRateSchedule warmup(int epochs, LearningRateSchedule then) {
        if (epochs < 1) {
            throw new IllegalArgumentException("epochs must be at least 1, got " + epochs);
        }
        return (epoch, base) -> epoch <= epochs ? base * epoch / epochs : then.rate(epoch, base);
    }

    /**
     * Like {@link #warmup(int, LearningRateSchedule)}, optionally with the following schedule
     * starting at its own first epoch when warm-up ends, so a cosine of {@code n} epochs
     * after {@code w} epochs of warm-up runs whole, ending at epoch {@code w + n}.
     *
     * @param epochs the warm-up length
     * @param then the schedule after warm-up
     * @param shift {@code true} to count {@code then} from the end of warm-up (epoch
     *     {@code epochs + 1} is its epoch 1), {@code false} from the first epoch
     * @return the schedule
     */
    static LearningRateSchedule warmup(int epochs, LearningRateSchedule then, boolean shift) {
        if (!shift) {
            return warmup(epochs, then);
        }
        return warmup(epochs, (epoch, base) -> then.rate(epoch - epochs, base));
    }

    /**
     * Lowers the rate in a straight line from the base to {@code minRate} over
     * {@code totalEpochs}, then stays there: the default of Hugging Face's trainer.
     *
     * @param totalEpochs the length of the decay
     * @param minRate the final rate
     * @return the schedule
     */
    static LearningRateSchedule linear(int totalEpochs, float minRate) {
        if (totalEpochs < 1) {
            throw new IllegalArgumentException("totalEpochs must be at least 1, got " + totalEpochs);
        }
        return (epoch, base) -> {
            double t = Math.min(1.0, (epoch - 1) / (double) Math.max(1, totalEpochs - 1));
            return (float) (base + (minRate - base) * t);
        };
    }

    /**
     * Warmup-stable-decay (Hu et al., 2024, MiniCPM), the schedule of many recent LLMs: a
     * linear rise to the base rate, a long stretch at it, then a short decay to
     * {@code minRate}. Training can stop or branch at any point of the stable stretch.
     *
     * @param warmup steps of the rise
     * @param stable steps at the base rate
     * @param decay steps of the decay; afterwards the rate stays at {@code minRate}
     * @param minRate the final rate
     * @param sqrtDecay {@code true} for the {@code 1 - sqrt(t)} cool-down shape, else linear
     * @return the schedule
     */
    static LearningRateSchedule warmupStableDecay(int warmup, int stable, int decay, float minRate, boolean sqrtDecay) {
        if (warmup < 0 || stable < 0 || decay < 1) {
            throw new IllegalArgumentException("need warmup >= 0, stable >= 0 and decay >= 1");
        }
        return (step, base) -> {
            if (step <= warmup) {
                return base * step / (float) warmup;
            }
            if (step <= warmup + stable) {
                return base;
            }
            double t = Math.min(1.0, (step - warmup - stable) / (double) decay);
            double f = sqrtDecay ? 1 - Math.sqrt(t) : 1 - t;
            return (float) (minRate + (base - minRate) * f);
        };
    }

    /**
     * Cosine annealing with warm restarts (Loshchilov and Hutter, 2017, SGDR): a cosine from
     * the base to {@code minRate} over {@code period} epochs, then again over a period
     * {@code multiplier} times longer, and so on.
     *
     * @param period the length of the first cycle
     * @param multiplier how much longer each next cycle is, at least 1
     * @param minRate the rate at the end of every cycle
     * @return the schedule
     */
    static LearningRateSchedule cosineWithRestarts(int period, float multiplier, float minRate) {
        if (period < 1 || !(multiplier >= 1)) {
            throw new IllegalArgumentException("need period >= 1 and multiplier >= 1");
        }
        return (epoch, base) -> {
            double t = epoch - 1;
            double length = period;
            while (t >= length) {
                t -= length;
                length *= multiplier;
            }
            return (float) (minRate + (base - minRate) * 0.5 * (1 + Math.cos(Math.PI * t / length)));
        };
    }

    /**
     * The 1cycle policy (Smith, 2018) as PyTorch's {@code OneCycleLR}: a cosine rise from
     * {@code base / 25} to the base over the first 30% of the steps, then a cosine fall to
     * {@code base / 25 / 10^4}.
     *
     * @param total the number of steps
     * @return the schedule
     */
    static LearningRateSchedule oneCycle(int total) {
        if (total < 2) {
            throw new IllegalArgumentException("total must be at least 2, got " + total);
        }
        double up = Math.max(1, 0.3 * total - 1);   // the step at the peak, as PyTorch counts
        return (epoch, base) -> {
            double start = base / 25.0;
            double end = start / 1e4;
            int s = Math.min(epoch - 1, total - 1);
            boolean rising = s <= up;
            double from = rising ? start : base;
            double to = rising ? base : end;
            double t = rising ? s / up : (s - up) / Math.max(1, total - 1 - up);
            return (float) (to + (from - to) * 0.5 * (1 + Math.cos(Math.PI * t)));
        };
    }

    /**
     * The original Transformer's schedule (Vaswani et al., 2017): a linear rise over
     * {@code warmup} steps, then decay with the inverse square root of the step, matching at
     * the peak.
     *
     * @param warmup steps of the rise
     * @return the schedule
     */
    static LearningRateSchedule inverseSqrt(int warmup) {
        if (warmup < 1) {
            throw new IllegalArgumentException("warmup must be at least 1, got " + warmup);
        }
        return (epoch, base) -> epoch <= warmup ? base * epoch / warmup
                : (float) (base * Math.sqrt(warmup / (double) epoch));
    }
}
