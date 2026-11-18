package io.github.zzuegg.brainiac.experiments.statistics;

import io.github.zzuegg.brainiac.experimental.stats.RankModels;
import io.github.zzuegg.brainiac.experimental.stats.Ratings;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Learning rankings from comparisons (item 12.1.14): Bradley-Terry by MM, Rank Centrality and
 * Plackett-Luce against the raw win rate on sparse Erdos-Renyi comparison graphs with planted
 * Bradley-Terry strengths; Borda, Kemeny local search and Plackett-Luce for aggregating noisy
 * full rankings; and TrueSkill and Glicko-2 against Elo on a match stream with player turnover
 * and skill drift, scored by next-match log loss overall and on newcomers' first ten games.
 */
public final class Comparisons {

    private Comparisons() {
    }

    /**
     * Ranking from sparse pairwise comparisons.
     *
     * @param items items
     * @param perItem comparisons per item on average
     * @param reps replications
     * @param seed the seed
     * @return mean Kendall tau with the true order per method
     */
    public static Map<String, Double> pairwise(int items, int perItem, int reps, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (int rep = 0; rep < reps; rep++) {
            SplittableRandom r = new SplittableRandom(seed + rep);
            double[] theta = new double[items];
            for (int i = 0; i < items; i++) {
                theta[i] = 1.2 * r.nextGaussian();
            }
            List<int[]> games = new ArrayList<>();
            int m = items * perItem / 2;
            for (int g = 0; g < m; g++) {
                int a = r.nextInt(items);
                int b = r.nextInt(items - 1);
                if (b >= a) {
                    b++;
                }
                double pa = 1 / (1 + Math.exp(theta[b] - theta[a]));
                games.add(r.nextDouble() < pa ? new int[] {a, b} : new int[] {b, a});
            }
            out.merge("win rate", Scores.kendall(RankModels.winRate(items, games), theta) / reps, Double::sum);
            for (RankModels.Method method : new RankModels.Method[] {RankModels.Method.BRADLEY_TERRY_MM,
                RankModels.Method.RANK_CENTRALITY, RankModels.Method.PLACKETT_LUCE}) {
                out.merge(method.name(), Scores.kendall(RankModels.fromPairs(method, items, games), theta) / reps,
                        Double::sum);
            }
        }
        return out;
    }

    /**
     * Aggregation of noisy full rankings drawn from a Plackett-Luce model.
     *
     * @param items items
     * @param voters rankings
     * @param seed the seed
     * @return Kendall tau with the true order per method
     */
    public static Map<String, Double> aggregation(int items, int voters, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] theta = new double[items];
        for (int i = 0; i < items; i++) {
            theta[i] = r.nextGaussian();
        }
        List<int[]> rankings = new ArrayList<>();
        for (int v = 0; v < voters; v++) {
            // Plackett-Luce sampling by Gumbel-perturbed sorting
            Integer[] o = new Integer[items];
            double[] key = new double[items];
            for (int i = 0; i < items; i++) {
                o[i] = i;
                key[i] = theta[i] - Math.log(-Math.log(r.nextDouble()));
            }
            java.util.Arrays.sort(o, (a, b) -> Double.compare(key[b], key[a]));
            int[] rk = new int[items];
            for (int i = 0; i < items; i++) {
                rk[i] = o[i];
            }
            rankings.add(rk);
        }
        Map<String, Double> out = new LinkedHashMap<>();
        for (RankModels.Method m : new RankModels.Method[] {RankModels.Method.BORDA,
            RankModels.Method.KEMENY_LOCAL_SEARCH, RankModels.Method.PLACKETT_LUCE}) {
            out.put(m.name(), Scores.kendall(RankModels.fromRankings(m, items, rankings), theta));
        }
        out.put("single voter", Scores.kendall(RankModels.fromRankings(RankModels.Method.BORDA, items,
                rankings.subList(0, 1)), theta));
        return out;
    }

    /**
     * A match stream with turnover and drift.
     *
     * @param players player slots
     * @param matches matches
     * @param turnoverEvery replace {@code turnover} players every this many matches
     * @param turnover players replaced each time
     * @param drift the per-match random-walk step of every skill
     * @param seed the seed
     * @return {@code {log loss overall, log loss on newcomers' first 10 games}} per system
     */
    public static Map<String, double[]> ratings(int players, int matches, int turnoverEvery, int turnover,
            double drift, long seed) {
        Map<String, Ratings.RatingSystem> systems = new LinkedHashMap<>();
        systems.put("Elo K=32", Ratings.elo(players, 32));
        systems.put("TrueSkill", Ratings.trueSkill(players));
        systems.put("Glicko-2", Ratings.glicko2(players));
        Map<String, double[]> loss = new LinkedHashMap<>();
        systems.keySet().forEach(k -> loss.put(k, new double[4]));
        SplittableRandom r = new SplittableRandom(seed);
        double[] skill = new double[players];
        int[] played = new int[players];
        for (int i = 0; i < players; i++) {
            skill[i] = r.nextGaussian() * 1.5;
        }
        boolean[] newcomer = new boolean[players];
        for (int t = 0; t < matches; t++) {
            if (t > 0 && t % turnoverEvery == 0) {
                for (int k = 0; k < turnover; k++) {
                    int i = r.nextInt(players);
                    skill[i] = r.nextGaussian() * 1.5;
                    played[i] = 0;
                    newcomer[i] = true;
                    for (Ratings.RatingSystem s : systems.values()) {
                        s.reset(i);
                    }
                }
            }
            for (int i = 0; i < players; i++) {
                skill[i] += drift * r.nextGaussian();
            }
            int a = r.nextInt(players);
            int b = r.nextInt(players - 1);
            if (b >= a) {
                b++;
            }
            boolean aWins = r.nextDouble() < 1 / (1 + Math.exp(skill[b] - skill[a]));
            boolean fresh = (newcomer[a] && played[a] < 10) || (newcomer[b] && played[b] < 10);
            for (Map.Entry<String, Ratings.RatingSystem> e : systems.entrySet()) {
                double p = Math.min(1 - 1e-9, Math.max(1e-9, e.getValue().predict(a, b)));
                double l = -Math.log(aWins ? p : 1 - p);
                double[] acc = loss.get(e.getKey());
                acc[0] += l;
                acc[1]++;
                if (fresh) {
                    acc[2] += l;
                    acc[3]++;
                }
                if (aWins) {
                    e.getValue().update(a, b);
                } else {
                    e.getValue().update(b, a);
                }
            }
            played[a]++;
            played[b]++;
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        loss.forEach((k, v) -> out.put(k, new double[] {v[0] / v[1], v[3] == 0 ? Double.NaN : v[2] / v[3]}));
        return out;
    }
}
