package io.github.zzuegg.brainiac.experimental.bio;

import io.github.zzuegg.brainiac.experimental.data.BioSequences;
import io.github.zzuegg.brainiac.experimental.data.BioSequences.Tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Phylogenetic inference on DNA alignments: Jukes-Cantor and Kimura two-parameter distances,
 * UPGMA (assumes a molecular clock) and neighbour joining (consistent for additive distances),
 * Fitch parsimony, Felsenstein's pruning likelihood under JC/K2P with per-branch length
 * optimisation, and maximum-likelihood search by nearest-neighbour interchange. Trees use the
 * rooted {@link Tree} representation of {@link BioSequences} (leaves first, root last); under a
 * reversible model the likelihood does not depend on the root position.
 *
 * <p>Sokal and Michener (1958); Saitou and Nei, "The neighbor-joining method" (MBE 1987);
 * Fitch (Syst. Zool. 1971); Felsenstein, "Evolutionary trees from DNA sequences: a maximum
 * likelihood approach" (J. Mol. Evol. 1981).
 */
public final class Phylogenetics {

    private Phylogenetics() {
    }

    /**
     * Pairwise evolutionary distances.
     *
     * @param alignment {@code [taxon][site]} in ACGT
     * @param kimura Kimura two-parameter when true, Jukes-Cantor otherwise
     * @return the symmetric distance matrix
     */
    public static double[][] distances(int[][] alignment, boolean kimura) {
        int n = alignment.length;
        int sites = alignment[0].length;
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int ts = 0;
                int tv = 0;
                for (int s = 0; s < sites; s++) {
                    int a = alignment[i][s];
                    int b = alignment[j][s];
                    if (a != b) {
                        if (a % 2 == b % 2) {
                            ts++;
                        } else {
                            tv++;
                        }
                    }
                }
                double p = ts / (double) sites;
                double q = tv / (double) sites;
                double v;
                if (kimura) {
                    double x = Math.max(1 - 2 * p - q, 1e-3);
                    double y = Math.max(1 - 2 * q, 1e-3);
                    v = -0.5 * Math.log(x) - 0.25 * Math.log(y);
                } else {
                    double pd = p + q;
                    v = -0.75 * Math.log(Math.max(1 - 4 * pd / 3, 1e-3));
                }
                d[i][j] = v;
                d[j][i] = v;
            }
        }
        return d;
    }

    /**
     * UPGMA: repeatedly joins the closest clusters, averaging distances by cluster size.
     *
     * @param dist the distance matrix
     * @return the ultrametric tree
     */
    public static Tree upgma(double[][] dist) {
        int n = dist.length;
        int nodes = 2 * n - 1;
        int[] parent = new int[nodes];
        double[] len = new double[nodes];
        double[] height = new double[nodes];
        double[][] d = new double[nodes][nodes];
        for (int i = 0; i < n; i++) {
            System.arraycopy(dist[i], 0, d[i], 0, n);
        }
        int[] size = new int[nodes];
        List<Integer> live = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            live.add(i);
            size[i] = 1;
        }
        for (int v = n; v < nodes; v++) {
            int ba = -1;
            int bb = -1;
            double best = Double.POSITIVE_INFINITY;
            for (int x = 0; x < live.size(); x++) {
                for (int y = x + 1; y < live.size(); y++) {
                    double dv = d[live.get(x)][live.get(y)];
                    if (dv < best) {
                        best = dv;
                        ba = live.get(x);
                        bb = live.get(y);
                    }
                }
            }
            height[v] = best / 2;
            parent[ba] = v;
            parent[bb] = v;
            len[ba] = Math.max(height[v] - height[ba], 1e-4);
            len[bb] = Math.max(height[v] - height[bb], 1e-4);
            size[v] = size[ba] + size[bb];
            live.remove(Integer.valueOf(ba));
            live.remove(Integer.valueOf(bb));
            for (int o : live) {
                double dv = (d[ba][o] * size[ba] + d[bb][o] * size[bb]) / size[v];
                d[v][o] = dv;
                d[o][v] = dv;
            }
            live.add(v);
        }
        parent[nodes - 1] = -1;
        return new Tree(parent, len, n);
    }

    /**
     * Neighbour joining with the Q-criterion; the last three clusters are joined at a root.
     *
     * @param dist the distance matrix
     * @return the tree (rooted at the last join)
     */
    public static Tree neighbourJoining(double[][] dist) {
        int n = dist.length;
        int nodes = 2 * n - 1;
        int[] parent = new int[nodes];
        double[] len = new double[nodes];
        double[][] d = new double[nodes][nodes];
        for (int i = 0; i < n; i++) {
            System.arraycopy(dist[i], 0, d[i], 0, n);
        }
        List<Integer> live = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            live.add(i);
        }
        int v = n;
        while (live.size() > 2) {
            int r = live.size();
            double[] sum = new double[nodes];
            for (int a : live) {
                for (int b : live) {
                    sum[a] += d[a][b];
                }
            }
            int ba = -1;
            int bb = -1;
            double best = Double.POSITIVE_INFINITY;
            for (int x = 0; x < r; x++) {
                for (int y = x + 1; y < r; y++) {
                    int a = live.get(x);
                    int b = live.get(y);
                    double q = (r - 2) * d[a][b] - sum[a] - sum[b];
                    if (q < best) {
                        best = q;
                        ba = a;
                        bb = b;
                    }
                }
            }
            double la = 0.5 * d[ba][bb] + (sum[ba] - sum[bb]) / (2.0 * (r - 2));
            double lb = d[ba][bb] - la;
            parent[ba] = v;
            parent[bb] = v;
            len[ba] = Math.max(la, 1e-4);
            len[bb] = Math.max(lb, 1e-4);
            live.remove(Integer.valueOf(ba));
            live.remove(Integer.valueOf(bb));
            for (int o : live) {
                double dv = 0.5 * (d[ba][o] + d[bb][o] - d[ba][bb]);
                d[v][o] = dv;
                d[o][v] = dv;
            }
            live.add(v);
            v++;
        }
        int a = live.get(0);
        int b = live.get(1);
        parent[a] = v;
        parent[b] = v;
        len[a] = Math.max(d[a][b] / 2, 1e-4);
        len[b] = Math.max(d[a][b] / 2, 1e-4);
        parent[v] = -1;
        return new Tree(parent, len, n);
    }

    /**
     * The Fitch parsimony score (minimum number of substitutions) of a tree.
     *
     * @param tree the tree
     * @param alignment {@code [taxon][site]}
     * @return the parsimony score
     */
    public static int parsimony(Tree tree, int[][] alignment) {
        int nodes = tree.parent().length;
        int sites = alignment[0].length;
        int[] order = postOrder(tree);
        int[][] set = new int[nodes][sites];
        List<List<Integer>> children = BioSequences.childLists(tree);
        int score = 0;
        for (int v : order) {
            if (v < tree.taxa()) {
                for (int s = 0; s < sites; s++) {
                    set[v][s] = 1 << alignment[v][s];
                }
                continue;
            }
            int c0 = children.get(v).get(0);
            int c1 = children.get(v).get(1);
            for (int s = 0; s < sites; s++) {
                int inter = set[c0][s] & set[c1][s];
                if (inter != 0) {
                    set[v][s] = inter;
                } else {
                    set[v][s] = set[c0][s] | set[c1][s];
                    score++;
                }
            }
        }
        return score;
    }

    /**
     * The nodes in post-order (children before parents).
     *
     * @param tree the tree
     * @return the order
     */
    public static int[] postOrder(Tree tree) {
        List<List<Integer>> children = BioSequences.childLists(tree);
        int[] out = new int[tree.parent().length];
        int[] idx = {0};
        visit(tree.root(), children, out, idx);
        return out;
    }

    private static void visit(int v, List<List<Integer>> children, int[] out, int[] idx) {
        for (int c : children.get(v)) {
            visit(c, children, out, idx);
        }
        out[idx[0]++] = v;
    }

    /**
     * Felsenstein's pruning log-likelihood under JC ({@code kappa = 1}) or K2P with uniform base
     * frequencies.
     *
     * @param tree the tree with branch lengths
     * @param alignment {@code [taxon][site]}
     * @param kappa the transition/transversion ratio
     * @return the log-likelihood
     */
    public static double logLikelihood(Tree tree, int[][] alignment, double kappa) {
        int nodes = tree.parent().length;
        int sites = alignment[0].length;
        int[] order = postOrder(tree);
        List<List<Integer>> children = BioSequences.childLists(tree);
        double[][][] partial = new double[nodes][][];
        double[] logScale = new double[sites];
        for (int v : order) {
            double[][] p = new double[sites][4];
            if (v < tree.taxa()) {
                for (int s = 0; s < sites; s++) {
                    p[s][alignment[v][s]] = 1;
                }
            } else {
                for (double[] row : p) {
                    Arrays.fill(row, 1);
                }
                for (int c : children.get(v)) {
                    double[][] m = BioSequences.k2pMatrix(tree.length()[c], kappa);
                    double[][] pc = partial[c];
                    for (int s = 0; s < sites; s++) {
                        for (int x = 0; x < 4; x++) {
                            double acc = 0;
                            for (int y = 0; y < 4; y++) {
                                acc += m[x][y] * pc[s][y];
                            }
                            p[s][x] *= acc;
                        }
                    }
                    partial[c] = null;
                }
                for (int s = 0; s < sites; s++) {
                    double mx = Math.max(Math.max(p[s][0], p[s][1]), Math.max(p[s][2], p[s][3]));
                    if (mx < 1e-100) {
                        mx = 1e-100;
                    }
                    for (int x = 0; x < 4; x++) {
                        p[s][x] /= mx;
                    }
                    logScale[s] += Math.log(mx);
                }
            }
            partial[v] = p;
        }
        double ll = 0;
        double[][] root = partial[tree.root()];
        for (int s = 0; s < sites; s++) {
            ll += Math.log(0.25 * (root[s][0] + root[s][1] + root[s][2] + root[s][3])) + logScale[s];
        }
        return ll;
    }

    /**
     * Optimises every branch length by golden-section search on the log-likelihood (in log
     * branch length), for a few rounds.
     *
     * @param tree the tree (topology kept)
     * @param alignment the alignment
     * @param kappa the transition/transversion ratio
     * @param rounds passes over all branches
     * @return the tree with optimised lengths
     */
    public static Tree optimiseBranches(Tree tree, int[][] alignment, double kappa, int rounds) {
        double[] len = tree.length().clone();
        for (int r = 0; r < rounds; r++) {
            for (int v = 0; v < len.length - 1; v++) {
                optimiseBranch(tree.parent(), len, tree.taxa(), v, alignment, kappa);
            }
        }
        return new Tree(tree.parent().clone(), len, tree.taxa());
    }

    private static void optimiseBranch(int[] parent, double[] len, int taxa, int v, int[][] alignment, double kappa) {
        double lo = Math.log(1e-4);
        double hi = Math.log(3);
        double g = (Math.sqrt(5) - 1) / 2;
        double a = hi - g * (hi - lo);
        double b = lo + g * (hi - lo);
        double fa = evalLen(parent, len, taxa, v, a, alignment, kappa);
        double fb = evalLen(parent, len, taxa, v, b, alignment, kappa);
        for (int it = 0; it < 14; it++) {
            if (fa > fb) {
                hi = b;
                b = a;
                fb = fa;
                a = hi - g * (hi - lo);
                fa = evalLen(parent, len, taxa, v, a, alignment, kappa);
            } else {
                lo = a;
                a = b;
                fa = fb;
                b = lo + g * (hi - lo);
                fb = evalLen(parent, len, taxa, v, b, alignment, kappa);
            }
        }
        len[v] = Math.exp((lo + hi) / 2);
    }

    private static double evalLen(int[] parent, double[] len, int taxa, int v, double logLen, int[][] alignment,
            double kappa) {
        double old = len[v];
        len[v] = Math.exp(logLen);
        double ll = logLikelihood(new Tree(parent, len, taxa), alignment, kappa);
        len[v] = old;
        return ll;
    }

    /**
     * Maximum-likelihood search by nearest-neighbour interchange: around every internal edge the
     * two alternative topologies are tried (with the five adjacent branches re-optimised) and the
     * best improvement is kept, until no interchange improves the likelihood.
     *
     * @param start the starting tree (usually neighbour joining)
     * @param alignment the alignment
     * @param kappa the transition/transversion ratio
     * @param maxRounds the maximum number of improving passes
     * @return the locally optimal tree
     */
    public static Tree nniSearch(Tree start, int[][] alignment, double kappa, int maxRounds) {
        Tree best = optimiseBranches(start, alignment, kappa, 2);
        double bestLl = logLikelihood(best, alignment, kappa);
        for (int round = 0; round < maxRounds; round++) {
            boolean improved = false;
            for (Tree cand : nniNeighbours(best)) {
                // re-optimise the branches the interchange touched: the two moved subtrees and
                // the internal edge between them
                double[] len = cand.length();
                for (int v = 0; v < len.length - 1; v++) {
                    if (cand.parent()[v] != best.parent()[v]) {
                        optimiseBranch(cand.parent(), len, cand.taxa(), v, alignment, kappa);
                        optimiseBranch(cand.parent(), len, cand.taxa(), cand.parent()[v], alignment, kappa);
                    }
                }
                Tree opt = cand;
                double ll = logLikelihood(opt, alignment, kappa);
                if (ll > bestLl + 1e-6) {
                    bestLl = ll;
                    best = opt;
                    improved = true;
                    break;
                }
            }
            if (!improved) {
                break;
            }
        }
        return best;
    }

    /**
     * All nearest-neighbour interchanges of a rooted binary tree: for each internal non-root node
     * {@code u} with children {@code a, b} and sibling {@code c}, swap {@code a} or {@code b}
     * with {@code c}.
     *
     * @param tree the tree
     * @return the neighbouring trees
     */
    public static List<Tree> nniNeighbours(Tree tree) {
        List<Tree> out = new ArrayList<>();
        List<List<Integer>> children = BioSequences.childLists(tree);
        int[] parent = tree.parent();
        for (int u = tree.taxa(); u < parent.length; u++) {
            int p = parent[u];
            if (p < 0) {
                continue;
            }
            int c = children.get(p).get(0) == u ? children.get(p).get(1) : children.get(p).get(0);
            for (int swap : children.get(u)) {
                int[] np = parent.clone();
                np[swap] = p;
                np[c] = u;
                out.add(new Tree(np, tree.length().clone(), tree.taxa()));
            }
        }
        return out;
    }
}
