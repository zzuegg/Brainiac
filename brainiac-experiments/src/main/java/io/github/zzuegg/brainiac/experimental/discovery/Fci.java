package io.github.zzuegg.brainiac.experimental.discovery;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Fast Causal Inference (Spirtes, Glymour and Scheines 2000; orientation rules of Zhang, AIJ
 * 2008, <a href="https://doi.org/10.1016/j.artint.2008.08.001">doi:10.1016/j.artint.2008.08.001</a>),
 * which drops PC's causal-sufficiency assumption and returns a partial ancestral graph (PAG).
 *
 * <ol>
 *   <li>The {@link PcAlgorithm#skeleton stable PC skeleton}.</li>
 *   <li>Unshielded colliders on a graph of circle marks.</li>
 *   <li>Possible-D-SEP removal: an edge {@code i - j} is also removed when the pair tests
 *       independent given a subset of {@code Possible-D-SEP(i)}, the nodes reachable from
 *       {@code i} by paths whose every inner node is a collider or sits in a triangle; this
 *       catches the separations that hidden confounders hide from neighbour sets.</li>
 *   <li>Colliders again from the final separating sets, then Zhang's rules R1 (away from
 *       colliders), R2 (away from cycles) and R3 (double triangle) until nothing changes.</li>
 * </ol>
 *
 * The discriminating-path rule R4 and the tail rules R5 to R10 (selection bias and the
 * completeness of tails) are not implemented, so some circles that complete FCI would orient stay
 * circles; every mark that is oriented is sound under the rules used.
 *
 * <p>Marks: {@code pag[i][j]} is the mark at {@code j} on the edge between {@code i} and
 * {@code j}: {@link #NONE}, {@link #CIRCLE}, {@link #ARROW} or {@link #TAIL}.
 */
public final class Fci {

    /** No edge. */
    public static final int NONE = 0;
    /** An undetermined mark. */
    public static final int CIRCLE = 1;
    /** An arrowhead: the node is not an ancestor of the other end. */
    public static final int ARROW = 2;
    /** A tail: the node is an ancestor of the other end. */
    public static final int TAIL = 3;

    private final double alpha;
    private final int maxConditioning;
    private final int maxPossibleDsep;

    /**
     * Creates the learner.
     *
     * @param alpha the CI-test level
     * @param maxConditioning the largest conditioning set
     * @param maxPossibleDsep the largest Possible-D-SEP set searched (larger sets are truncated)
     */
    public Fci(double alpha, int maxConditioning, int maxPossibleDsep) {
        this.alpha = alpha;
        this.maxConditioning = maxConditioning;
        this.maxPossibleDsep = maxPossibleDsep;
    }

    /**
     * Runs FCI.
     *
     * @param test the CI test
     * @param d the number of variables
     * @return the PAG marks {@code [d][d]}
     */
    public int[][] fit(CiTest test, int d) {
        PcAlgorithm.Skeleton sk = PcAlgorithm.skeleton(test, d, alpha, maxConditioning);
        boolean[][] adj = sk.adjacent();
        int[][][] sep = sk.sepsets();
        int[][] m = circles(adj);
        colliders(m, adj, sep);
        // Possible-D-SEP stage
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                if (i == j || !adj[i][j]) {
                    continue;
                }
                List<Integer> pds = new ArrayList<>(possibleDsep(m, adj, i, j));
                if (pds.size() > maxPossibleDsep) {
                    pds = pds.subList(0, maxPossibleDsep);
                }
                boolean removed = false;
                for (int l = 1; l <= Math.min(maxConditioning, pds.size()) && !removed; l++) {
                    for (int[] s : PcAlgorithm.subsets(pds, l)) {
                        if (test.pValue(i, j, s) > alpha) {
                            adj[i][j] = false;
                            adj[j][i] = false;
                            sep[i][j] = s;
                            sep[j][i] = s;
                            removed = true;
                            break;
                        }
                    }
                }
            }
        }
        m = circles(adj);
        colliders(m, adj, sep);
        rules(m);
        return m;
    }

    private static int[][] circles(boolean[][] adj) {
        int d = adj.length;
        int[][] m = new int[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                m[i][j] = adj[i][j] ? CIRCLE : NONE;
            }
        }
        return m;
    }

    private static void colliders(int[][] m, boolean[][] adj, int[][][] sep) {
        int d = adj.length;
        for (int k = 0; k < d; k++) {
            for (int i = 0; i < d; i++) {
                for (int j = i + 1; j < d; j++) {
                    if (adj[i][k] && adj[j][k] && !adj[i][j] && !PcAlgorithm.contains(sep[i][j], k)) {
                        m[i][k] = ARROW;
                        m[j][k] = ARROW;
                    }
                }
            }
        }
    }

    private static Set<Integer> possibleDsep(int[][] m, boolean[][] adj, int i, int j) {
        int d = adj.length;
        Set<Integer> out = new LinkedHashSet<>();
        boolean[][] seen = new boolean[d][d];
        Deque<int[]> queue = new ArrayDeque<>();
        for (int k = 0; k < d; k++) {
            if (adj[i][k]) {
                seen[i][k] = true;
                queue.add(new int[] {i, k});
                if (k != j) {
                    out.add(k);
                }
            }
        }
        while (!queue.isEmpty()) {
            int[] e = queue.poll();
            int a = e[0];
            int b = e[1];
            for (int c = 0; c < d; c++) {
                if (c == a || c == i || !adj[b][c] || seen[b][c]) {
                    continue;
                }
                boolean collider = m[a][b] == ARROW && m[c][b] == ARROW;
                if (collider || adj[a][c]) {
                    seen[b][c] = true;
                    queue.add(new int[] {b, c});
                    if (c != j) {
                        out.add(c);
                    }
                }
            }
        }
        return out;
    }

    private static boolean adjacent(int[][] m, int a, int b) {
        return m[a][b] != NONE;
    }

    private static void rules(int[][] m) {
        int d = m.length;
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    if (a == b || !adjacent(m, a, b)) {
                        continue;
                    }
                    for (int c = 0; c < d; c++) {
                        if (c == a || c == b || !adjacent(m, b, c)) {
                            continue;
                        }
                        // R1: a *-> b o-* c, a and c not adjacent  =>  b -> c
                        if (m[a][b] == ARROW && m[c][b] == CIRCLE && !adjacent(m, a, c)) {
                            m[c][b] = TAIL;
                            m[b][c] = ARROW;
                            changed = true;
                        }
                        // R2: (a -> b *-> c or a *-> b -> c) and a *-o c  =>  a *-> c
                        if (adjacent(m, a, c) && m[a][c] == CIRCLE) {
                            boolean first = m[a][b] == ARROW && m[b][a] == TAIL && m[b][c] == ARROW;
                            boolean second = m[a][b] == ARROW && m[b][c] == ARROW && m[c][b] == TAIL;
                            if (first || second) {
                                m[a][c] = ARROW;
                                changed = true;
                            }
                        }
                    }
                }
            }
            // R3: a *-> b <-* c, a *-o t o-* c, a and c not adjacent, t *-o b  =>  t *-> b
            for (int b = 0; b < d; b++) {
                for (int t = 0; t < d; t++) {
                    if (t == b || !adjacent(m, t, b) || m[t][b] != CIRCLE) {
                        continue;
                    }
                    boolean fire = false;
                    for (int a = 0; a < d && !fire; a++) {
                        for (int c = a + 1; c < d && !fire; c++) {
                            if (a == b || c == b || a == t || c == t || adjacent(m, a, c)) {
                                continue;
                            }
                            fire = m[a][b] == ARROW && m[c][b] == ARROW && adjacent(m, a, t) && adjacent(m, c, t)
                                    && m[a][t] == CIRCLE && m[c][t] == CIRCLE;
                        }
                    }
                    if (fire) {
                        m[t][b] = ARROW;
                        changed = true;
                    }
                }
            }
        }
    }
}
