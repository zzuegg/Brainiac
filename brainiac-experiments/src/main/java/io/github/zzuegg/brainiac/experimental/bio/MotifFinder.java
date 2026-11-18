package io.github.zzuegg.brainiac.experimental.bio;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Motif discovery in unaligned sequences.
 *
 * <ul>
 *   <li>{@link Method#MEME_EM}: MEME's expectation maximisation for a two-component mixture
 *       (motif position weight matrix vs background) under the ZOOPS model (zero or one
 *       occurrence per sequence), started from the best of many substrings of the data
 *       (Bailey and Elkan, ISMB 1994).</li>
 *   <li>{@link Method#GIBBS_SAMPLER}: the site sampler of Lawrence et al. (Science 1993): each
 *       step holds one sequence out, builds the PWM from the other sequences' sites and samples
 *       the held-out site in proportion to its likelihood ratio; restarts keep the best
 *       information content.</li>
 *   <li>{@link Method#KMER_ENRICHMENT}: the counting baseline: the {@code W}-mer most
 *       over-represented in the sequences against shuffled-background expectation, turned into
 *       a smoothed consensus PWM.</li>
 * </ul>
 */
public final class MotifFinder {

    /** A motif discovery method. */
    public enum Method {
        /** EM with zero or one occurrence per sequence. */
        MEME_EM,
        /** the Gibbs site sampler. */
        GIBBS_SAMPLER,
        /** the most enriched k-mer as a consensus PWM. */
        KMER_ENRICHMENT
    }

    /**
     * A discovered motif.
     *
     * @param pwm {@code [width][alphabet]} probabilities
     * @param sites the predicted site start in each sequence, or -1
     */
    public record Motif(double[][] pwm, int[] sites) {
    }

    private MotifFinder() {
    }

    /**
     * Finds a motif.
     *
     * @param method the method
     * @param sequences the sequences (usually the positives)
     * @param width the motif width
     * @param alphabet the alphabet size
     * @param seed the seed
     * @return the motif
     */
    public static Motif find(Method method, int[][] sequences, int width, int alphabet, long seed) {
        double[] bg = background(sequences, alphabet);
        return switch (method) {
            case MEME_EM -> meme(sequences, width, bg, seed);
            case GIBBS_SAMPLER -> gibbs(sequences, width, bg, seed);
            case KMER_ENRICHMENT -> kmer(sequences, width, bg);
        };
    }

    private static double[] background(int[][] sequences, int alphabet) {
        double[] bg = new double[alphabet];
        double total = 0;
        for (int[] s : sequences) {
            for (int x : s) {
                bg[x]++;
                total++;
            }
        }
        for (int a = 0; a < alphabet; a++) {
            bg[a] = (bg[a] + 1) / (total + alphabet);
        }
        return bg;
    }

    private static double logRatio(double[][] logPwm, double[] logBg, int[] s, int start) {
        double v = 0;
        for (int w = 0; w < logPwm.length; w++) {
            v += logPwm[w][s[start + w]] - logBg[s[start + w]];
        }
        return v;
    }

    private static double[][] log(double[][] p) {
        double[][] out = new double[p.length][];
        for (int i = 0; i < p.length; i++) {
            out[i] = new double[p[i].length];
            for (int a = 0; a < p[i].length; a++) {
                out[i][a] = Math.log(p[i][a]);
            }
        }
        return out;
    }

    private static double[] log(double[] p) {
        double[] out = new double[p.length];
        for (int a = 0; a < p.length; a++) {
            out[a] = Math.log(p[a]);
        }
        return out;
    }

    private static Motif meme(int[][] seqs, int width, double[] bg, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        int alphabet = bg.length;
        double bestLl = Double.NEGATIVE_INFINITY;
        double[][] bestPwm = null;
        double bestGamma = 0.5;
        // starting points: substrings of random sequences, each refined by a few EM steps
        int starts = Math.min(60, seqs.length * 2);
        for (int st = 0; st < starts; st++) {
            int[] s = seqs[rnd.nextInt(seqs.length)];
            int p = rnd.nextInt(s.length - width + 1);
            double[][] pwm = new double[width][alphabet];
            for (int w = 0; w < width; w++) {
                for (int a = 0; a < alphabet; a++) {
                    pwm[w][a] = a == s[p + w] ? 0.5 : 0.5 / (alphabet - 1);
                }
            }
            double[] gamma = {0.5};
            double ll = 0;
            for (int it = 0; it < 3; it++) {
                ll = emStep(seqs, pwm, bg, gamma, null);
            }
            if (ll > bestLl) {
                bestLl = ll;
                bestPwm = pwm;
                bestGamma = gamma[0];
            }
        }
        double[] gamma = {bestGamma};
        double[][] z = new double[seqs.length][];
        for (int it = 0; it < 40; it++) {
            emStep(seqs, bestPwm, bg, gamma, z);
        }
        emStep(seqs, bestPwm, bg, gamma, z);
        int[] sites = new int[seqs.length];
        for (int i = 0; i < seqs.length; i++) {
            int arg = -1;
            double best = 0.5;
            for (int j = 0; j < z[i].length; j++) {
                if (z[i][j] > best) {
                    best = z[i][j];
                    arg = j;
                }
            }
            sites[i] = arg;
        }
        return new Motif(bestPwm, sites);
    }

    /** One ZOOPS EM step in place; returns the log-likelihood of the data before the update. */
    private static double emStep(int[][] seqs, double[][] pwm, double[] bg, double[] gamma, double[][] zOut) {
        int width = pwm.length;
        int alphabet = bg.length;
        double[][] lp = log(pwm);
        double[] lb = log(bg);
        double[][] counts = new double[width][alphabet];
        double ll = 0;
        double zTotal = 0;
        for (int i = 0; i < seqs.length; i++) {
            int[] s = seqs[i];
            int m = s.length - width + 1;
            double[] lr = new double[m];
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < m; j++) {
                lr[j] = logRatio(lp, lb, s, j);
                max = Math.max(max, lr[j]);
            }
            double sum = 0;
            for (int j = 0; j < m; j++) {
                sum += Math.exp(lr[j] - max);
            }
            // P(x | site somewhere) / P(x | background) = (1/m) sum LR
            double motifTerm = Math.log(gamma[0] / m) + max + Math.log(sum);
            double nullTerm = Math.log(1 - gamma[0]);
            double denom = Math.max(motifTerm, nullTerm)
                    + Math.log(Math.exp(motifTerm - Math.max(motifTerm, nullTerm))
                            + Math.exp(nullTerm - Math.max(motifTerm, nullTerm)));
            ll += denom;
            double[] z = new double[m];
            for (int j = 0; j < m; j++) {
                z[j] = Math.exp(Math.log(gamma[0] / m) + lr[j] - denom);
                zTotal += z[j];
                for (int w = 0; w < width; w++) {
                    counts[w][s[j + w]] += z[j];
                }
            }
            if (zOut != null) {
                zOut[i] = z;
            }
        }
        for (int w = 0; w < width; w++) {
            double tot = 0;
            for (int a = 0; a < alphabet; a++) {
                tot += counts[w][a] + 0.25 * bg[a] * alphabet / 4;
            }
            for (int a = 0; a < alphabet; a++) {
                pwm[w][a] = (counts[w][a] + 0.25 * bg[a] * alphabet / 4) / tot;
            }
        }
        gamma[0] = Math.min(0.99, Math.max(0.01, zTotal / seqs.length));
        return ll;
    }

    private static Motif gibbs(int[][] seqs, int width, double[] bg, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        int alphabet = bg.length;
        double[] lb = log(bg);
        double bestScore = Double.NEGATIVE_INFINITY;
        int[] bestSites = null;
        for (int restart = 0; restart < 6; restart++) {
            int[] sites = new int[seqs.length];
            for (int i = 0; i < seqs.length; i++) {
                sites[i] = rnd.nextInt(seqs[i].length - width + 1);
            }
            int[][] counts = new int[width][alphabet];
            for (int i = 0; i < seqs.length; i++) {
                add(counts, seqs[i], sites[i], 1);
            }
            for (int it = 0; it < 60 * seqs.length; it++) {
                int z = rnd.nextInt(seqs.length);
                add(counts, seqs[z], sites[z], -1);
                double[][] lp = new double[width][alphabet];
                for (int w = 0; w < width; w++) {
                    for (int a = 0; a < alphabet; a++) {
                        lp[w][a] = Math.log((counts[w][a] + bg[a]) / (seqs.length - 1 + 1.0));
                    }
                }
                int m = seqs[z].length - width + 1;
                double[] lr = new double[m];
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < m; j++) {
                    lr[j] = logRatio(lp, lb, seqs[z], j);
                    max = Math.max(max, lr[j]);
                }
                double sum = 0;
                for (int j = 0; j < m; j++) {
                    lr[j] = Math.exp(lr[j] - max);
                    sum += lr[j];
                }
                double u = rnd.nextDouble() * sum;
                int pick = m - 1;
                for (int j = 0; j < m; j++) {
                    u -= lr[j];
                    if (u <= 0) {
                        pick = j;
                        break;
                    }
                }
                sites[z] = pick;
                add(counts, seqs[z], pick, 1);
            }
            double score = 0;
            double[][] pwm = toPwm(counts, bg, seqs.length);
            for (int w = 0; w < width; w++) {
                for (int a = 0; a < alphabet; a++) {
                    score += counts[w][a] * Math.log(pwm[w][a] / bg[a]);
                }
            }
            if (score > bestScore) {
                bestScore = score;
                bestSites = sites.clone();
            }
        }
        int[][] counts = new int[width][alphabet];
        for (int i = 0; i < seqs.length; i++) {
            add(counts, seqs[i], bestSites[i], 1);
        }
        double[][] pwm = toPwm(counts, bg, seqs.length);
        // ZOOPS read-out: keep only sites whose log-odds are positive
        double[][] lp = log(pwm);
        int[] sites = bestSites.clone();
        for (int i = 0; i < seqs.length; i++) {
            if (logRatio(lp, lb, seqs[i], sites[i]) < 0) {
                sites[i] = -1;
            }
        }
        return new Motif(pwm, sites);
    }

    private static double[][] toPwm(int[][] counts, double[] bg, int n) {
        double[][] pwm = new double[counts.length][bg.length];
        for (int w = 0; w < counts.length; w++) {
            for (int a = 0; a < bg.length; a++) {
                pwm[w][a] = (counts[w][a] + bg[a]) / (n + 1.0);
            }
        }
        return pwm;
    }

    private static void add(int[][] counts, int[] s, int start, int delta) {
        for (int w = 0; w < counts.length; w++) {
            counts[w][s[start + w]] += delta;
        }
    }

    private static Motif kmer(int[][] seqs, int width, double[] bg) {
        Map<Long, Integer> counts = new HashMap<>();
        int alphabet = bg.length;
        for (int[] s : seqs) {
            for (int j = 0; j + width <= s.length; j++) {
                counts.merge(code(s, j, width, alphabet), 1, Integer::sum);
            }
        }
        double[] lb = log(bg);
        long best = 0;
        double bestZ = Double.NEGATIVE_INFINITY;
        long total = 0;
        for (int[] s : seqs) {
            total += Math.max(0, s.length - width + 1);
        }
        for (Map.Entry<Long, Integer> e : counts.entrySet()) {
            long c = e.getKey();
            double logP = 0;
            long code = c;
            for (int w = 0; w < width; w++) {
                logP += lb[(int) (code % alphabet)];
                code /= alphabet;
            }
            double expected = total * Math.exp(logP);
            double z = (e.getValue() - expected) / Math.sqrt(expected + 1);
            if (z > bestZ) {
                bestZ = z;
                best = c;
            }
        }
        int[] consensus = new int[width];
        long code = best;
        for (int w = width - 1; w >= 0; w--) {
            consensus[w] = (int) (code % alphabet);
            code /= alphabet;
        }
        double[][] pwm = new double[width][alphabet];
        for (int w = 0; w < width; w++) {
            Arrays.fill(pwm[w], 0.3 / (alphabet - 1));
            pwm[w][consensus[w]] = 0.7;
        }
        int[] sites = new int[seqs.length];
        for (int i = 0; i < seqs.length; i++) {
            sites[i] = -1;
            for (int j = 0; j + width <= seqs[i].length; j++) {
                int mism = 0;
                for (int w = 0; w < width && mism <= 1; w++) {
                    mism += seqs[i][j + w] != consensus[w] ? 1 : 0;
                }
                if (mism <= 1) {
                    sites[i] = j;
                    break;
                }
            }
        }
        return new Motif(pwm, sites);
    }

    private static long code(int[] s, int start, int width, int alphabet) {
        long c = 0;
        for (int w = 0; w < width; w++) {
            c = c * alphabet + s[start + w];
        }
        return c;
    }

    /**
     * The mean Pearson correlation between the columns of a found PWM and the true PWM, at the
     * best relative shift (at least half the columns overlapping).
     *
     * @param found the found PWM
     * @param truth the true PWM
     * @return the mean column correlation, in [-1, 1]
     */
    public static double pwmCorrelation(double[][] found, double[][] truth) {
        double best = -1;
        int w = truth.length;
        for (int shift = -w / 2; shift <= w / 2; shift++) {
            double total = 0;
            int cols = 0;
            for (int c = 0; c < w; c++) {
                int f = c + shift;
                if (f < 0 || f >= found.length) {
                    continue;
                }
                total += pearson(found[f], truth[c]);
                cols++;
            }
            if (cols * 2 >= w) {
                best = Math.max(best, total / w);
            }
        }
        return best;
    }

    private static double pearson(double[] a, double[] b) {
        double ma = Arrays.stream(a).average().orElse(0);
        double mb = Arrays.stream(b).average().orElse(0);
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < a.length; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return saa * sbb == 0 ? 0 : sab / Math.sqrt(saa * sbb);
    }

    /**
     * Site-level F1: a predicted site is a hit if it overlaps the true site by at least half the
     * width.
     *
     * @param predicted predicted starts (-1 for none)
     * @param truth true starts (-1 for none)
     * @param width the motif width
     * @return the F1 score
     */
    public static double siteF1(int[] predicted, int[] truth, int width) {
        int tp = 0;
        int fp = 0;
        int fn = 0;
        for (int i = 0; i < truth.length; i++) {
            boolean hit = predicted[i] >= 0 && truth[i] >= 0 && Math.abs(predicted[i] - truth[i]) * 2 <= width;
            if (hit) {
                tp++;
            } else {
                fp += predicted[i] >= 0 ? 1 : 0;
                fn += truth[i] >= 0 ? 1 : 0;
            }
        }
        return tp == 0 ? 0 : 2.0 * tp / (2.0 * tp + fp + fn);
    }
}
