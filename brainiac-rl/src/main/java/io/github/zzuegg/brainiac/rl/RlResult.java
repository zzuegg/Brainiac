package io.github.zzuegg.brainiac.rl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** The episodes of a training run. */
public final class RlResult {

    private final List<Episode> episodes = new ArrayList<>();
    private long steps;

    /** Creates an empty result. */
    public RlResult() {
    }

    void add(Episode e) {
        episodes.add(e);
    }

    void steps(long total) {
        this.steps = total;
    }

    /** @return every finished episode, oldest first */
    public List<Episode> episodes() {
        return Collections.unmodifiableList(episodes);
    }

    /** @return the environment steps taken */
    public long totalSteps() {
        return steps;
    }

    /**
     * @param last how many of the latest episodes to average
     * @return their mean total reward, or {@code NaN} with no episodes
     */
    public double meanReturn(int last) {
        int n = Math.min(last, episodes.size());
        if (n == 0) {
            return Double.NaN;
        }
        double sum = 0;
        for (int i = episodes.size() - n; i < episodes.size(); i++) {
            sum += episodes.get(i).totalReward();
        }
        return sum / n;
    }

    /**
     * @param window how many episodes each point averages
     * @return the moving average of total reward, one value per episode from the
     *     {@code window}-th on
     */
    public double[] movingAverage(int window) {
        int n = episodes.size() - window + 1;
        if (n <= 0) {
            return new double[0];
        }
        double[] out = new double[n];
        double sum = 0;
        for (int i = 0; i < episodes.size(); i++) {
            sum += episodes.get(i).totalReward();
            if (i >= window) {
                sum -= episodes.get(i - window).totalReward();
            }
            if (i >= window - 1) {
                out[i - window + 1] = sum / window;
            }
        }
        return out;
    }

    /** @return the highest total reward of any episode, or {@code NaN} with none */
    public double bestReturn() {
        return episodes.stream().mapToDouble(Episode::totalReward).max().orElse(Double.NaN);
    }

    @Override
    public String toString() {
        return episodes.size() + " episodes, " + steps + " steps, mean return of the last 20 "
                + String.format("%.2f", meanReturn(20));
    }
}
