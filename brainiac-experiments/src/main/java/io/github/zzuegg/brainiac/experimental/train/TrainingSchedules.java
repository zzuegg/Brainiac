package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.train.LearningRateSchedule;

/**
 * Learning-rate schedules missing from {@link LearningRateSchedule}, as drop-in
 * {@code LearningRateSchedule}s ({@code epoch} is whatever unit the trainer counts: epochs for
 * {@code Trainer}, steps for {@code TrainingLoop}).
 *
 * <ul>
 *   <li>{@link #warmupStableDecay}: WSD, linear warm-up, a long constant phase, then a short
 *       decay. Unlike cosine it need not know the run length in advance: any checkpoint of
 *       the stable phase can be cooled down into a finished model, so one run yields models
 *       for many budgets (MiniCPM, DeepSeek, Llama 3 annealing). Hu et al., "MiniCPM",
 *       <a href="https://arxiv.org/abs/2404.06395">arXiv:2404.06395</a>; Hägele et al.,
 *       <a href="https://arxiv.org/abs/2405.18392">arXiv:2405.18392</a>.</li>
 *   <li>{@link #cosineWithRestarts}: SGDR, cosine cycles that restart at the base rate, each
 *       cycle {@code multiplier} times longer. Loshchilov and Hutter,
 *       <a href="https://arxiv.org/abs/1608.03983">arXiv:1608.03983</a>.</li>
 *   <li>{@link #oneCycle}: 1cycle, a cosine rise from {@code base / divFactor} to the base
 *       and a long cosine fall far below it, which trains small models in few epochs
 *       ("super-convergence"). Smith and Topin,
 *       <a href="https://arxiv.org/abs/1708.07120">arXiv:1708.07120</a>.</li>
 * </ul>
 */
public final class TrainingSchedules {

    private TrainingSchedules() {
    }

    /**
     * Warmup-stable-decay with a linear decay.
     *
     * @param warmup steps of linear warm-up from near 0 to the base rate
     * @param stable steps at the base rate
     * @param decay steps of decay to {@code minRate}; afterwards the rate stays there
     * @param minRate the final rate
     * @return the schedule
     */
    public static LearningRateSchedule warmupStableDecay(int warmup, int stable, int decay, float minRate) {
        return warmupStableDecay(warmup, stable, decay, minRate, false);
    }

    /**
     * Warmup-stable-decay.
     *
     * @param warmup steps of linear warm-up
     * @param stable steps at the base rate
     * @param decay steps of decay to {@code minRate}
     * @param minRate the final rate
     * @param sqrtDecay {@code true} for the "1 - sqrt" cool-down shape, which the scaling
     *     study found slightly better than linear
     * @return the schedule
     */
    public static LearningRateSchedule warmupStableDecay(int warmup, int stable, int decay, float minRate,
            boolean sqrtDecay) {
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
     * Cosine annealing with warm restarts (SGDR).
     *
     * @param period the length of the first cycle
     * @param multiplier how much longer each next cycle is, at least 1
     * @param minRate the rate at the end of every cycle
     * @return the schedule
     */
    public static LearningRateSchedule cosineWithRestarts(int period, float multiplier, float minRate) {
        if (period < 1 || multiplier < 1) {
            throw new IllegalArgumentException("need period >= 1 and multiplier >= 1");
        }
        return (step, base) -> {
            double t = step - 1;
            double length = period;
            while (t >= length) {
                t -= length;
                length *= multiplier;
            }
            return (float) (minRate + (base - minRate) * 0.5 * (1 + Math.cos(Math.PI * t / length)));
        };
    }

    /**
     * The 1cycle policy with the usual constants: 30% rise, start at {@code base / 25}, end
     * at {@code base / 25 / 10^4}.
     *
     * @param total the total number of steps
     * @return the schedule
     */
    public static LearningRateSchedule oneCycle(int total) {
        return oneCycle(total, 0.3f, 25f, 1e4f);
    }

    /**
     * The 1cycle policy; the base rate is the peak.
     *
     * @param total the total number of steps
     * @param rise the share of steps spent rising
     * @param divFactor the start rate is {@code base / divFactor}
     * @param finalDivFactor the end rate is {@code base / divFactor / finalDivFactor}
     * @return the schedule
     */
    public static LearningRateSchedule oneCycle(int total, float rise, float divFactor, float finalDivFactor) {
        if (total < 2 || !(rise > 0 && rise < 1)) {
            throw new IllegalArgumentException("need total >= 2 and rise in (0, 1)");
        }
        int up = Math.max(1, Math.round(total * rise));
        return (step, base) -> {
            double start = base / divFactor;
            double end = start / finalDivFactor;
            int s = Math.min(step - 1, total - 1);
            if (s < up) {
                return (float) cos(start, base, s / (double) up);
            }
            return (float) cos(base, end, (s - up) / (double) Math.max(1, total - 1 - up));
        };
    }

    private static double cos(double from, double to, double t) {
        return to + (from - to) * 0.5 * (1 + Math.cos(Math.PI * t));
    }
}
