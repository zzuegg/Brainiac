package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.experimental.llm.IrtEvaluator;

/**
 * Label models for weak supervision: they combine the votes of noisy labelling functions (or
 * crowd annotators) into probabilistic labels without ground truth, from the functions'
 * agreements and disagreements. Votes are {@code [rows][functions]} class ids, {@code -1} for
 * abstain. The probabilistic labels then train an end model with a soft (noise-aware) loss.
 *
 * <ul>
 *   <li>{@link #majorityVote}: the baseline.</li>
 *   <li>{@link #dawidSkene} (Dawid and Skene, 1979, <a href="https://doi.org/10.2307/2346806">doi:10.2307/2346806</a>):
 *       a full confusion matrix per function, fitted by EM.</li>
 *   <li>{@link #dataProgramming} (Ratner et al., NeurIPS 2016,
 *       <a href="https://arxiv.org/abs/1605.07723">arXiv:1605.07723</a>): the generative factor
 *       model with one accuracy per function (right with probability {@code a_j} when it votes,
 *       otherwise a uniformly random wrong class), fitted by EM on the marginal likelihood. Given
 *       correlated pairs, each pair is modelled by one joint factor over its two outcomes (abstain,
 *       right, wrong; both wrong the same or differently), so a duplicated heuristic is not
 *       counted twice.</li>
 *   <li>{@link #matrixCompletion} (the Snorkel MeTaL estimator; Ratner et al., AAAI 2019,
 *       <a href="https://arxiv.org/abs/1810.02840">arXiv:1810.02840</a>): with conditionally
 *       independent functions, the centred agreement rate of two functions factorises as
 *       {@code mu_i mu_j} with {@code mu = (K a - 1) / (K - 1)}, so each accuracy follows in closed
 *       form from triplets, {@code mu_i = sqrt(M_ij M_ik / M_jk)}, averaged over pairs.</li>
 *   <li>{@link #irtWeighted}: the item-response model of {@link IrtEvaluator} fitted to whether
 *       each function agrees with the majority, used as a weighting baseline.</li>
 * </ul>
 */
public final class LabelModel {

    private LabelModel() {
    }

    /**
     * A fitted label model.
     *
     * @param posteriors probabilistic labels, {@code [rows][classes]}
     * @param accuracies estimated accuracy of every function where it votes
     */
    public record Fit(double[][] posteriors, double[] accuracies) {

        /**
         * @param votes the votes the model was fitted to
         * @param truth the true classes
         * @return the accuracy of the posteriors' argmax on rows with at least one vote
         */
        public double coveredAccuracy(int[][] votes, int[] truth) {
            int n = 0;
            int right = 0;
            for (int i = 0; i < truth.length; i++) {
                boolean covered = false;
                for (int v : votes[i]) {
                    covered |= v >= 0;
                }
                if (covered) {
                    n++;
                    right += argmax(posteriors[i]) == truth[i] ? 1 : 0;
                }
            }
            return right / (double) Math.max(1, n);
        }
    }

    /**
     * Majority vote, ties split evenly, uncovered rows uniform.
     *
     * @param votes the votes
     * @param classes the number of classes
     * @return the vote shares (accuracies NaN)
     */
    public static Fit majorityVote(int[][] votes, int classes) {
        double[] weights = new double[votes[0].length];
        java.util.Arrays.fill(weights, 1);
        double[][] p = weightedVote(votes, classes, weights, true);
        double[] acc = new double[weights.length];
        java.util.Arrays.fill(acc, Double.NaN);
        return new Fit(p, acc);
    }

    private static double[][] weightedVote(int[][] votes, int classes, double[] weights, boolean hard) {
        double[][] p = new double[votes.length][classes];
        for (int i = 0; i < votes.length; i++) {
            double[] s = new double[classes];
            for (int j = 0; j < votes[i].length; j++) {
                if (votes[i][j] >= 0) {
                    s[votes[i][j]] += weights[j];
                }
            }
            if (hard) {
                double max = Double.NEGATIVE_INFINITY;
                for (double v : s) {
                    max = Math.max(max, v);
                }
                int ties = 0;
                for (double v : s) {
                    ties += v == max ? 1 : 0;
                }
                for (int c = 0; c < classes; c++) {
                    p[i][c] = s[c] == max ? 1.0 / ties : 0;
                }
            } else {
                double max = Double.NEGATIVE_INFINITY;
                for (double v : s) {
                    max = Math.max(max, v);
                }
                double z = 0;
                for (int c = 0; c < classes; c++) {
                    p[i][c] = Math.exp(s[c] - max);
                    z += p[i][c];
                }
                for (int c = 0; c < classes; c++) {
                    p[i][c] /= z;
                }
            }
        }
        return p;
    }

    /**
     * Dawid-Skene EM, started from majority vote.
     *
     * @param votes the votes
     * @param classes the number of classes
     * @param iterations EM iterations
     * @return the fit (accuracies: the confusion matrices' class-weighted diagonals)
     */
    public static Fit dawidSkene(int[][] votes, int classes, int iterations) {
        int n = votes.length;
        int m = votes[0].length;
        double[][] q = majorityVote(votes, classes).posteriors();
        double[][][] conf = new double[m][classes][classes];
        double[] prior = new double[classes];
        for (int it = 0; it < iterations; it++) {
            java.util.Arrays.fill(prior, 1e-3);
            for (double[][] cj : conf) {
                for (double[] row : cj) {
                    java.util.Arrays.fill(row, 1e-2);
                }
            }
            for (int i = 0; i < n; i++) {
                for (int y = 0; y < classes; y++) {
                    prior[y] += q[i][y];
                    for (int j = 0; j < m; j++) {
                        if (votes[i][j] >= 0) {
                            conf[j][y][votes[i][j]] += q[i][y];
                        }
                    }
                }
            }
            normalise(prior);
            for (double[][] cj : conf) {
                for (double[] row : cj) {
                    normalise(row);
                }
            }
            for (int i = 0; i < n; i++) {
                double[] lp = new double[classes];
                for (int y = 0; y < classes; y++) {
                    lp[y] = Math.log(prior[y]);
                    for (int j = 0; j < m; j++) {
                        if (votes[i][j] >= 0) {
                            lp[y] += Math.log(conf[j][y][votes[i][j]]);
                        }
                    }
                }
                q[i] = softmax(lp);
            }
        }
        double[] acc = new double[m];
        for (int j = 0; j < m; j++) {
            for (int y = 0; y < classes; y++) {
                acc[j] += prior[y] * conf[j][y][y];
            }
        }
        return new Fit(q, acc);
    }

    /**
     * The data-programming factor model.
     *
     * @param votes the votes
     * @param classes the number of classes
     * @param iterations EM iterations
     * @param correlatedPairs pairs of function indices modelled jointly (may be empty)
     * @return the fit
     */
    public static Fit dataProgramming(int[][] votes, int classes, int iterations, int[][] correlatedPairs) {
        int n = votes.length;
        int m = votes[0].length;
        boolean[] paired = new boolean[m];
        for (int[] p : correlatedPairs) {
            paired[p[0]] = true;
            paired[p[1]] = true;
        }
        double[] acc = new double[m];
        java.util.Arrays.fill(acc, 0.7);
        double[] prior = new double[classes];
        java.util.Arrays.fill(prior, 1.0 / classes);
        // joint outcome tables of the pairs: 0 abstain, 1 right, 2 wrong for each, plus 9 = both
        // wrong with the same class (8 then means both wrong, different classes)
        double[][] table = new double[correlatedPairs.length][10];
        for (double[] t : table) {
            java.util.Arrays.fill(t, 0.1);
        }
        double[][] q = new double[n][classes];
        for (int it = 0; it <= iterations; it++) {
            for (int i = 0; i < n; i++) {
                double[] lp = new double[classes];
                for (int y = 0; y < classes; y++) {
                    lp[y] = Math.log(prior[y]);
                    for (int j = 0; j < m; j++) {
                        int v = votes[i][j];
                        if (v >= 0 && !paired[j]) {
                            lp[y] += v == y ? Math.log(acc[j]) : Math.log((1 - acc[j]) / (classes - 1));
                        }
                    }
                    for (int p = 0; p < correlatedPairs.length; p++) {
                        int cell = pairCell(votes[i][correlatedPairs[p][0]], votes[i][correlatedPairs[p][1]], y);
                        lp[y] += Math.log(table[p][cell] / pairMultiplicity(cell, classes));
                    }
                }
                q[i] = softmax(lp);
            }
            if (it == iterations) {
                break;
            }
            double[] right = new double[m];
            double[] cast = new double[m];
            java.util.Arrays.fill(prior, 1e-3);
            for (double[] t : table) {
                java.util.Arrays.fill(t, 1e-2);
            }
            for (int i = 0; i < n; i++) {
                for (int y = 0; y < classes; y++) {
                    prior[y] += q[i][y];
                    for (int p = 0; p < correlatedPairs.length; p++) {
                        table[p][pairCell(votes[i][correlatedPairs[p][0]], votes[i][correlatedPairs[p][1]], y)] +=
                                q[i][y];
                    }
                }
                for (int j = 0; j < m; j++) {
                    if (votes[i][j] >= 0) {
                        cast[j]++;
                        right[j] += q[i][votes[i][j]];
                    }
                }
            }
            normalise(prior);
            for (double[] t : table) {
                normalise(t);
            }
            for (int j = 0; j < m; j++) {
                acc[j] = Math.clamp((right[j] + 1) / (cast[j] + 2), 1e-3, 1 - 1e-3);
            }
        }
        return new Fit(q, acc);
    }

    private static int pairCell(int a, int b, int y) {
        int ca = a < 0 ? 0 : a == y ? 1 : 2;
        int cb = b < 0 ? 0 : b == y ? 1 : 2;
        if (ca == 2 && cb == 2 && a == b) {
            return 9;
        }
        return ca * 3 + cb;
    }

    private static double pairMultiplicity(int cell, int classes) {
        if (cell == 9) {
            return classes - 1;
        }
        if (cell == 8) {
            return (classes - 1) * Math.max(1, classes - 2);
        }
        int ca = cell / 3;
        int cb = cell % 3;
        return (ca == 2 ? classes - 1 : 1) * (cb == 2 ? classes - 1 : 1);
    }

    /**
     * The closed-form triplet (matrix-completion) estimator of the accuracies, then a naive-Bayes
     * posterior with them.
     *
     * @param votes the votes
     * @param classes the number of classes
     * @return the fit
     */
    public static Fit matrixCompletion(int[][] votes, int classes) {
        int m = votes[0].length;
        double[][] mu = new double[m][m];
        for (int a = 0; a < m; a++) {
            for (int b = a + 1; b < m; b++) {
                int both = 0;
                int agree = 0;
                for (int[] v : votes) {
                    if (v[a] >= 0 && v[b] >= 0) {
                        both++;
                        agree += v[a] == v[b] ? 1 : 0;
                    }
                }
                double pAgree = both == 0 ? 1.0 / classes : agree / (double) both;
                mu[a][b] = (classes * pAgree - 1) / (classes - 1);
                mu[b][a] = mu[a][b];
            }
        }
        double[] acc = new double[m];
        for (int i = 0; i < m; i++) {
            double sum = 0;
            int count = 0;
            for (int j = 0; j < m; j++) {
                for (int k = j + 1; k < m; k++) {
                    if (j == i || k == i || Math.abs(mu[j][k]) < 1e-3) {
                        continue;
                    }
                    double v = mu[i][j] * mu[i][k] / mu[j][k];
                    if (v > 0) {
                        sum += Math.sqrt(v);
                        count++;
                    }
                }
            }
            double mui = count == 0 ? 0.4 : Math.clamp(sum / count, 0, 0.999);
            acc[i] = Math.clamp((mui * (classes - 1) + 1) / classes, 1.0 / classes + 1e-3, 0.999);
        }
        return new Fit(naiveBayes(votes, classes, acc), acc);
    }

    /**
     * IRT weighting: each row is a "respondent", each function an "item" answered correctly when
     * it agrees with the majority; the fitted item parameters give each function's agreement
     * probability at average ability, divided by its coverage, as an accuracy estimate.
     *
     * @param votes the votes
     * @param classes the number of classes
     * @return the fit
     */
    public static Fit irtWeighted(int[][] votes, int classes) {
        int n = votes.length;
        int m = votes[0].length;
        double[][] mv = majorityVote(votes, classes).posteriors();
        boolean[][] responses = new boolean[n][m];
        double[] coverage = new double[m];
        for (int i = 0; i < n; i++) {
            int major = argmax(mv[i]);
            for (int j = 0; j < m; j++) {
                responses[i][j] = votes[i][j] >= 0 && votes[i][j] == major;
                coverage[j] += votes[i][j] >= 0 ? 1.0 / n : 0;
            }
        }
        IrtEvaluator irt = IrtEvaluator.fit(responses, 300);
        double[] acc = new double[m];
        for (int j = 0; j < m; j++) {
            double p = 1 / (1 + Math.exp(-irt.discrimination()[j] * (0 - irt.difficulty()[j])));
            acc[j] = Math.clamp(p / Math.max(coverage[j], 1e-3), 1.0 / classes + 1e-3, 0.999);
        }
        return new Fit(naiveBayes(votes, classes, acc), acc);
    }

    /**
     * The posterior of independent functions with known accuracies and a uniform prior.
     *
     * @param votes the votes
     * @param classes the number of classes
     * @param accuracy each function's accuracy
     * @return the posteriors
     */
    public static double[][] naiveBayes(int[][] votes, int classes, double[] accuracy) {
        double[] w = new double[accuracy.length];
        for (int j = 0; j < w.length; j++) {
            w[j] = Math.log(accuracy[j] * (classes - 1) / (1 - accuracy[j]));
        }
        return weightedVote(votes, classes, w, false);
    }

    private static void normalise(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        for (int i = 0; i < v.length; i++) {
            v[i] /= s;
        }
    }

    private static double[] softmax(double[] lp) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : lp) {
            max = Math.max(max, v);
        }
        double[] out = new double[lp.length];
        double z = 0;
        for (int i = 0; i < lp.length; i++) {
            out[i] = Math.exp(lp[i] - max);
            z += out[i];
        }
        for (int i = 0; i < lp.length; i++) {
            out[i] /= z;
        }
        return out;
    }

    static int argmax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }
}
