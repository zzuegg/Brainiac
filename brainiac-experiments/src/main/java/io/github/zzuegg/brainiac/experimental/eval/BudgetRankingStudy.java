package io.github.zzuegg.brainiac.experimental.eval;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Do method rankings survive the training budget? Trains every method of a lineup on a budget
 * ladder (1x, 2x, 4x, ... the test budget) with several seeds, ranks the methods by their
 * seed-mean score at each budget, and compares each ranking with the ranking at the largest
 * budget by Kendall's τ, with a percentile bootstrap interval over seeds. For every pair of
 * methods it also finds the crossover budget: the smallest ladder budget from which the pair's
 * order agrees with its order at the largest budget at every later budget.
 *
 * <p>With a constant learning rate, one run per method and seed that logs its score at every
 * ladder budget is the same as separate runs of each budget, which is how {@link CurveRunner}s
 * are meant to be written; the logged curves also feed {@link CurveExtrapolator}. With a decaying
 * schedule each budget needs its own run.
 *
 * <p>Context: Zeng et al., "Are Transformers Effective for Time Series Forecasting?",
 * <a href="https://arxiv.org/abs/2205.13504">arXiv:2205.13504</a>, ranked transformers
 * against a linear map at one budget; this study makes the ranking against budget the measured
 * quantity.
 */
public final class BudgetRankingStudy {

    private BudgetRankingStudy() {
    }

    /** Trains one method with one seed and reports its score at every checkpoint. */
    @FunctionalInterface
    public interface CurveRunner {
        /**
         * Runs a method.
         *
         * @param method the method's name
         * @param seed the seed
         * @param checkpoints increasing training budgets (steps)
         * @return the score after each checkpoint's budget
         */
        double[] run(String method, long seed, int[] checkpoints);
    }

    /**
     * A pair whose order changes with the budget.
     *
     * @param winner the method ahead at the largest budget
     * @param loser the method behind at the largest budget
     * @param budget the smallest ladder budget from which the final order always holds
     * @param leaderAtSmallest the method ahead at the smallest budget
     */
    public record Crossover(String winner, String loser, int budget, String leaderAtSmallest) {
    }

    /**
     * The study's outcome.
     *
     * @param methods the methods
     * @param budgets the ladder
     * @param curves score per {@code [method][seed][budget]}
     * @param mean seed-mean score per {@code [method][budget]}
     * @param tau Kendall's τ between the ranking at each budget and at the largest
     * @param tauLow lower end of the 95% bootstrap interval of each τ
     * @param tauHigh upper end of the 95% bootstrap interval of each τ
     * @param crossovers every pair whose order at the smallest budget differs from the largest
     * @param lowerIsBetter whether lower scores rank first
     */
    public record Result(List<String> methods, int[] budgets, double[][][] curves, double[][] mean, double[] tau,
            double[] tauLow, double[] tauHigh, List<Crossover> crossovers, boolean lowerIsBetter) {

        /**
         * The methods ordered best first at one ladder position.
         *
         * @param budgetIndex the index into {@link #budgets}
         * @return method names, best first
         */
        public List<String> ranking(int budgetIndex) {
            Integer[] order = new Integer[methods.size()];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> lowerIsBetter
                    ? Double.compare(mean[a][budgetIndex], mean[b][budgetIndex])
                    : Double.compare(mean[b][budgetIndex], mean[a][budgetIndex]));
            List<String> out = new ArrayList<>();
            for (int i : order) {
                out.add(methods.get(i));
            }
            return out;
        }

        /**
         * The seed-mean score of a method at a ladder position.
         *
         * @param method the method
         * @param budgetIndex the index into {@link #budgets}
         * @return its mean score
         */
        public double score(String method, int budgetIndex) {
            return mean[methods.indexOf(method)][budgetIndex];
        }
    }

    /**
     * The budget ladder {@code base * 2^k}, {@code k = 0..doublings}.
     *
     * @param base the test budget
     * @param doublings how many times to double it
     * @return the ladder
     */
    public static int[] ladder(int base, int doublings) {
        int[] b = new int[doublings + 1];
        for (int k = 0; k <= doublings; k++) {
            b[k] = base << k;
        }
        return b;
    }

    /**
     * Runs every method and seed and analyses the curves.
     *
     * @param methods the lineup
     * @param budgets the ladder, increasing
     * @param seeds the seeds
     * @param lowerIsBetter whether lower scores rank first (losses) or higher (accuracies)
     * @param runner trains one method and seed
     * @param bootstrap bootstrap resamples for the τ intervals
     * @param seed the bootstrap seed
     * @return the result
     */
    public static Result run(List<String> methods, int[] budgets, long[] seeds, boolean lowerIsBetter,
            CurveRunner runner, int bootstrap, long seed) {
        double[][][] curves = new double[methods.size()][seeds.length][];
        for (int m = 0; m < methods.size(); m++) {
            for (int s = 0; s < seeds.length; s++) {
                curves[m][s] = runner.run(methods.get(m), seeds[s], budgets);
            }
        }
        return analyse(methods, budgets, curves, lowerIsBetter, bootstrap, seed);
    }

    /**
     * Analyses logged curves.
     *
     * @param methods the lineup
     * @param budgets the ladder, increasing
     * @param curves score per {@code [method][seed][budget]}
     * @param lowerIsBetter whether lower scores rank first
     * @param bootstrap bootstrap resamples (over seeds, per method) for the τ intervals
     * @param seed the bootstrap seed
     * @return the result
     */
    public static Result analyse(List<String> methods, int[] budgets, double[][][] curves, boolean lowerIsBetter,
            int bootstrap, long seed) {
        int m = methods.size();
        int nb = budgets.length;
        double[][] mean = seedMeans(curves, null);
        double[] tau = new double[nb];
        for (int b = 0; b < nb; b++) {
            tau[b] = kendallTau(column(mean, b), column(mean, nb - 1));
        }
        double[] low = new double[nb];
        double[] high = new double[nb];
        SplittableRandom r = new SplittableRandom(seed);
        double[][] samples = new double[nb][bootstrap];
        for (int k = 0; k < bootstrap; k++) {
            int[][] pick = new int[m][];
            for (int i = 0; i < m; i++) {
                pick[i] = new int[curves[i].length];
                for (int s = 0; s < pick[i].length; s++) {
                    pick[i][s] = r.nextInt(curves[i].length);
                }
            }
            double[][] bm = seedMeans(curves, pick);
            for (int b = 0; b < nb; b++) {
                samples[b][k] = kendallTau(column(bm, b), column(bm, nb - 1));
            }
        }
        for (int b = 0; b < nb; b++) {
            if (bootstrap > 0) {
                Arrays.sort(samples[b]);
                low[b] = samples[b][(int) Math.floor(0.025 * (bootstrap - 1))];
                high[b] = samples[b][(int) Math.ceil(0.975 * (bootstrap - 1))];
            } else {
                low[b] = tau[b];
                high[b] = tau[b];
            }
        }
        List<Crossover> crossovers = new ArrayList<>();
        double sign = lowerIsBetter ? -1 : 1;
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                double fin = Math.signum(sign * (mean[i][nb - 1] - mean[j][nb - 1]));
                double first = Math.signum(sign * (mean[i][0] - mean[j][0]));
                if (fin == 0 || first == fin) {
                    continue;
                }
                int from = nb - 1;
                while (from > 0 && Math.signum(sign * (mean[i][from - 1] - mean[j][from - 1])) == fin) {
                    from--;
                }
                String winner = fin > 0 ? methods.get(i) : methods.get(j);
                String loser = fin > 0 ? methods.get(j) : methods.get(i);
                String leader = first > 0 ? methods.get(i) : first < 0 ? methods.get(j) : "tie";
                crossovers.add(new Crossover(winner, loser, budgets[from], leader));
            }
        }
        return new Result(List.copyOf(methods), budgets.clone(), curves, mean, tau, low, high,
                List.copyOf(crossovers), lowerIsBetter);
    }

    private static double[][] seedMeans(double[][][] curves, int[][] pick) {
        double[][] mean = new double[curves.length][];
        for (int i = 0; i < curves.length; i++) {
            int nb = curves[i][0].length;
            mean[i] = new double[nb];
            int n = curves[i].length;
            for (int s = 0; s < n; s++) {
                double[] c = curves[i][pick == null ? s : pick[i][s]];
                for (int b = 0; b < nb; b++) {
                    mean[i][b] += c[b] / n;
                }
            }
        }
        return mean;
    }

    private static double[] column(double[][] m, int b) {
        double[] c = new double[m.length];
        for (int i = 0; i < m.length; i++) {
            c[i] = m[i][b];
        }
        return c;
    }

    /**
     * Kendall's τ-b between two score vectors (ties count as neither concordant nor
     * discordant, and the denominator is corrected for them).
     *
     * @param x the first scores
     * @param y the second scores, same length
     * @return τ in [-1, 1]; 0 when either vector is constant
     */
    public static double kendallTau(double[] x, double[] y) {
        long concordant = 0;
        long discordant = 0;
        long tiesX = 0;
        long tiesY = 0;
        for (int i = 0; i < x.length; i++) {
            for (int j = i + 1; j < x.length; j++) {
                double a = Math.signum(x[i] - x[j]);
                double b = Math.signum(y[i] - y[j]);
                if (a == 0 && b == 0) {
                    continue;
                }
                if (a == 0) {
                    tiesX++;
                } else if (b == 0) {
                    tiesY++;
                } else if (a == b) {
                    concordant++;
                } else {
                    discordant++;
                }
            }
        }
        double d = Math.sqrt((double) (concordant + discordant + tiesX) * (concordant + discordant + tiesY));
        return d == 0 ? 0 : (concordant - discordant) / d;
    }

    /**
     * Spearman's rank correlation (average ranks for ties).
     *
     * @param x the first scores
     * @param y the second scores, same length
     * @return ρ in [-1, 1]; 0 when either vector is constant
     */
    public static double spearman(double[] x, double[] y) {
        double[] rx = ranks(x);
        double[] ry = ranks(y);
        double mx = 0;
        double my = 0;
        for (int i = 0; i < x.length; i++) {
            mx += rx[i] / x.length;
            my += ry[i] / x.length;
        }
        double sxy = 0;
        double sxx = 0;
        double syy = 0;
        for (int i = 0; i < x.length; i++) {
            sxy += (rx[i] - mx) * (ry[i] - my);
            sxx += (rx[i] - mx) * (rx[i] - mx);
            syy += (ry[i] - my) * (ry[i] - my);
        }
        return sxx == 0 || syy == 0 ? 0 : sxy / Math.sqrt(sxx * syy);
    }

    /**
     * Average ranks, 1-based, ascending.
     *
     * @param x the scores
     * @return the rank of each score
     */
    public static double[] ranks(double[] x) {
        Integer[] order = new Integer[x.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(x[a], x[b]));
        double[] r = new double[x.length];
        int i = 0;
        while (i < order.length) {
            int j = i;
            while (j + 1 < order.length && x[order[j + 1]] == x[order[i]]) {
                j++;
            }
            for (int k = i; k <= j; k++) {
                r[order[k]] = (i + j) / 2.0 + 1;
            }
            i = j + 1;
        }
        return r;
    }

    /**
     * Formats a result as a table: one row per method with its mean score at each budget,
     * then τ with its interval per budget.
     *
     * @param r the result
     * @return the table
     */
    public static String table(Result r) {
        StringBuilder sb = new StringBuilder(String.format("%-20s", "method"));
        for (int b : r.budgets()) {
            sb.append(String.format(" %9s", b));
        }
        sb.append('\n');
        for (int i = 0; i < r.methods().size(); i++) {
            sb.append(String.format("%-20s", r.methods().get(i)));
            for (int b = 0; b < r.budgets().length; b++) {
                sb.append(String.format(" %9.4f", r.mean()[i][b]));
            }
            sb.append('\n');
        }
        sb.append(String.format("%-20s", "tau vs largest"));
        for (int b = 0; b < r.budgets().length; b++) {
            sb.append(String.format(" %9.2f", r.tau()[b]));
        }
        sb.append('\n').append(String.format("%-20s", "95% interval"));
        for (int b = 0; b < r.budgets().length; b++) {
            sb.append(String.format(" %4.1f:%4.1f", r.tauLow()[b], r.tauHigh()[b]));
        }
        return sb.append('\n').toString();
    }
}
