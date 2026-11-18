package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Vietoris-Rips persistent homology in dimensions 0 and 1 (Edelsbrunner and Harer 2010) and the
 * standard vectorisations of a barcode (persistence images, Adams et al. 2017,
 * <a href="https://arxiv.org/abs/1507.06217">arXiv:1507.06217</a>; Betti curves).
 *
 * <p>The filtration adds every edge at its length and every triangle at its longest edge.
 * {@code H0} comes from Kruskal's union-find: an edge that merges two components kills one (born
 * at 0). {@code H1} comes from the reduction over {@code F2} of the triangle boundary matrix:
 * each edge that closes a cycle gives birth to a loop, and the triangle whose reduced boundary
 * has that edge as its lowest entry kills it. Every pair keeps the vertices of the edges that
 * created and destroyed it, so each birth and death is a specific pairwise distance and
 * gradients can be routed to those points (see {@code experimental.nn.TopologicalLoss}).
 */
public final class PersistentHomology {

    private PersistentHomology() {
    }

    /**
     * One persistence pair.
     *
     * @param dimension 0 (component) or 1 (loop)
     * @param birth the birth radius (0 for components)
     * @param death the death radius ({@code Double.POSITIVE_INFINITY} if it never dies)
     * @param birthEdge the vertices {@code {i, j}} of the creating edge ({@code null} in dimension 0)
     * @param deathEdge the vertices {@code {i, j}} of the edge whose length is the death
     *     ({@code null} if it never dies)
     */
    public record Pair(int dimension, double birth, double death, int[] birthEdge, int[] deathEdge) {

        /** @return {@code death - birth} */
        public double persistence() {
            return death - birth;
        }
    }

    /**
     * A barcode.
     *
     * @param h0 the finite component pairs ({@code n - 1} for a connected filtration)
     * @param h1 the loop pairs
     */
    public record Barcode(List<Pair> h0, List<Pair> h1) {
    }

    private record Edge(int i, int j, double length) {
    }

    private static Edge[] sortedEdges(double[][] d, double maxRadius) {
        int n = d.length;
        List<Edge> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (d[i][j] <= maxRadius) {
                    edges.add(new Edge(i, j, d[i][j]));
                }
            }
        }
        Edge[] e = edges.toArray(new Edge[0]);
        Arrays.sort(e, (a, b) -> Double.compare(a.length, b.length));
        return e;
    }

    private static int find(int[] parent, int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    /**
     * The 0-dimensional pairs: the minimum spanning tree edges, each killing a component.
     *
     * @param d the distance matrix {@code [n][n]}
     * @return the {@code n - 1} finite pairs in order of death
     */
    public static List<Pair> h0(double[][] d) {
        List<Pair> out = new ArrayList<>();
        int[] parent = new int[d.length];
        for (int i = 0; i < parent.length; i++) {
            parent[i] = i;
        }
        for (Edge e : sortedEdges(d, Double.POSITIVE_INFINITY)) {
            int a = find(parent, e.i);
            int b = find(parent, e.j);
            if (a != b) {
                parent[a] = b;
                out.add(new Pair(0, 0, e.length, null, new int[] {e.i, e.j}));
            }
        }
        return out;
    }

    /**
     * Euclidean distance matrix.
     *
     * @param x the points {@code [n][d]}
     * @return {@code [n][n]}
     */
    public static double[][] distances(double[][] x) {
        double[][] d = MlMatrices.squaredDistances(x);
        for (double[] row : d) {
            for (int j = 0; j < row.length; j++) {
                row[j] = Math.sqrt(Math.max(0, row[j]));
            }
        }
        return d;
    }

    /**
     * The Rips barcode of a point cloud.
     *
     * @param x the points {@code [n][d]}
     * @param maxRadius the largest edge length in the filtration; loops alive there never die
     * @return the barcode
     */
    public static Barcode rips(double[][] x, double maxRadius) {
        return ripsFromDistances(distances(x), maxRadius);
    }

    /**
     * The Rips barcode of a finite metric space.
     *
     * @param d the distance matrix {@code [n][n]}
     * @param maxRadius the largest edge length in the filtration
     * @return the barcode
     */
    public static Barcode ripsFromDistances(double[][] d, double maxRadius) {
        int n = d.length;
        Edge[] edges = sortedEdges(d, maxRadius);
        int[][] index = new int[n][n];
        for (int[] row : index) {
            Arrays.fill(row, -1);
        }
        for (int k = 0; k < edges.length; k++) {
            index[edges[k].i][edges[k].j] = k;
            index[edges[k].j][edges[k].i] = k;
        }
        List<Pair> h0 = new ArrayList<>();
        boolean[] positive = new boolean[edges.length];
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
        for (int k = 0; k < edges.length; k++) {
            int a = find(parent, edges[k].i);
            int b = find(parent, edges[k].j);
            if (a != b) {
                parent[a] = b;
                h0.add(new Pair(0, 0, edges[k].length, null, new int[] {edges[k].i, edges[k].j}));
            } else {
                positive[k] = true;
            }
        }
        // triangles, each as its three edge indices sorted descending (the first is its value)
        List<int[]> triangles = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int ij = index[i][j];
                if (ij < 0) {
                    continue;
                }
                for (int k = j + 1; k < n; k++) {
                    int ik = index[i][k];
                    int jk = index[j][k];
                    if (ik >= 0 && jk >= 0) {
                        int[] t = {ij, ik, jk};
                        Arrays.sort(t);
                        triangles.add(new int[] {t[2], t[1], t[0]});
                    }
                }
            }
        }
        triangles.sort((p, q) -> {
            for (int c = 0; c < 3; c++) {
                if (p[c] != q[c]) {
                    return Integer.compare(p[c], q[c]);
                }
            }
            return 0;
        });
        Map<Integer, int[]> pivots = new HashMap<>();
        List<Pair> h1 = new ArrayList<>();
        boolean[] killed = new boolean[edges.length];
        for (int[] tri : triangles) {
            int[] col = tri;
            while (col.length > 0) {
                int[] other = pivots.get(col[0]);
                if (other == null) {
                    break;
                }
                col = symmetricDifference(col, other);
            }
            if (col.length == 0) {
                continue;
            }
            int low = col[0];
            pivots.put(low, col);
            Edge birth = edges[low];
            Edge death = edges[tri[0]];
            killed[low] = true;
            if (death.length > birth.length) {
                h1.add(new Pair(1, birth.length, death.length, new int[] {birth.i, birth.j},
                        new int[] {death.i, death.j}));
            }
        }
        for (int k = 0; k < edges.length; k++) {
            if (positive[k] && !killed[k]) {
                h1.add(new Pair(1, edges[k].length, Double.POSITIVE_INFINITY, new int[] {edges[k].i, edges[k].j},
                        null));
            }
        }
        return new Barcode(h0, h1);
    }

    /** Symmetric difference of two descending-sorted index arrays. */
    private static int[] symmetricDifference(int[] a, int[] b) {
        int[] out = new int[a.length + b.length];
        int i = 0;
        int j = 0;
        int k = 0;
        while (i < a.length || j < b.length) {
            if (j == b.length || (i < a.length && a[i] > b[j])) {
                out[k++] = a[i++];
            } else if (i == a.length || b[j] > a[i]) {
                out[k++] = b[j++];
            } else {
                i++;
                j++;
            }
        }
        return Arrays.copyOf(out, k);
    }

    /**
     * A persistence image: each pair, placed at {@code (birth, persistence)}, adds a Gaussian
     * weighted by its persistence, evaluated on a grid. Infinite deaths are capped at
     * {@code maxBirth + maxPersistence}.
     *
     * @param pairs the pairs of one dimension
     * @param resolution the grid side
     * @param maxBirth the birth range {@code [0, maxBirth]}
     * @param maxPersistence the persistence range {@code [0, maxPersistence]}
     * @param sigma the Gaussian width
     * @return {@code resolution * resolution} values, row = persistence, column = birth
     */
    public static double[] persistenceImage(List<Pair> pairs, int resolution, double maxBirth,
            double maxPersistence, double sigma) {
        double[] img = new double[resolution * resolution];
        for (Pair p : pairs) {
            double pers = Math.min(p.persistence(), maxPersistence);
            for (int r = 0; r < resolution; r++) {
                double py = (r + 0.5) * maxPersistence / resolution;
                for (int c = 0; c < resolution; c++) {
                    double px = (c + 0.5) * maxBirth / resolution;
                    double dx = px - p.birth();
                    double dy = py - pers;
                    img[r * resolution + c] += pers * Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                }
            }
        }
        return img;
    }

    /**
     * The Betti curve: the number of pairs alive at each of {@code bins} radii in
     * {@code [0, maxRadius]}.
     *
     * @param pairs the pairs of one dimension
     * @param maxRadius the largest radius
     * @param bins the number of radii
     * @return the counts
     */
    public static double[] bettiCurve(List<Pair> pairs, double maxRadius, int bins) {
        double[] out = new double[bins];
        for (int b = 0; b < bins; b++) {
            double t = (b + 0.5) * maxRadius / bins;
            for (Pair p : pairs) {
                if (p.birth() <= t && t < p.death()) {
                    out[b]++;
                }
            }
        }
        return out;
    }

    /**
     * The bottleneck distance between two diagrams: the smallest {@code eps} such that there is
     * a matching of the points, each allowed to go to the diagonal, moving none by more than
     * {@code eps} in the L-infinity norm. Infinite deaths should be capped first.
     *
     * @param a the first diagram
     * @param b the second diagram
     * @return the distance
     */
    public static double bottleneckDistance(List<Pair> a, List<Pair> b) {
        int n = a.size();
        int m = b.size();
        List<Double> candidates = new ArrayList<>();
        candidates.add(0.0);
        for (Pair p : a) {
            candidates.add(p.persistence() / 2);
            for (Pair q : b) {
                candidates.add(Math.max(Math.abs(p.birth() - q.birth()), Math.abs(p.death() - q.death())));
            }
        }
        for (Pair q : b) {
            candidates.add(q.persistence() / 2);
        }
        double[] c = candidates.stream().mapToDouble(Double::doubleValue).sorted().distinct().toArray();
        int lo = 0;
        int hi = c.length - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (perfectMatching(a, b, n, m, c[mid])) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return c[lo];
    }

    /** Left: a's points then b's diagonal copies; right: b's points then a's diagonal copies. */
    private static boolean perfectMatching(List<Pair> a, List<Pair> b, int n, int m, double eps) {
        int size = n + m;
        boolean[][] allowed = new boolean[size][size];
        double tol = eps + 1e-12;
        for (int i = 0; i < n; i++) {
            Pair p = a.get(i);
            for (int j = 0; j < m; j++) {
                Pair q = b.get(j);
                allowed[i][j] = Math.max(Math.abs(p.birth() - q.birth()), Math.abs(p.death() - q.death())) <= tol;
            }
            allowed[i][m + i] = p.persistence() / 2 <= tol;
        }
        for (int j = 0; j < m; j++) {
            allowed[n + j][j] = b.get(j).persistence() / 2 <= tol;
            for (int i = 0; i < n; i++) {
                allowed[n + j][m + i] = true;
            }
        }
        int[] matchRight = new int[size];
        Arrays.fill(matchRight, -1);
        for (int u = 0; u < size; u++) {
            if (!augment(u, allowed, matchRight, new boolean[size])) {
                return false;
            }
        }
        return true;
    }

    private static boolean augment(int u, boolean[][] allowed, int[] matchRight, boolean[] seen) {
        for (int v = 0; v < allowed[u].length; v++) {
            if (allowed[u][v] && !seen[v]) {
                seen[v] = true;
                if (matchRight[v] < 0 || augment(matchRight[v], allowed, matchRight, seen)) {
                    matchRight[v] = u;
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Caps infinite deaths at a radius, for {@link #bottleneckDistance} and images.
     *
     * @param pairs the pairs
     * @param radius the cap
     * @return the capped pairs
     */
    public static List<Pair> cap(List<Pair> pairs, double radius) {
        List<Pair> out = new ArrayList<>();
        for (Pair p : pairs) {
            out.add(Double.isInfinite(p.death())
                    ? new Pair(p.dimension(), p.birth(), Math.max(radius, p.birth()), p.birthEdge(), p.deathEdge())
                    : p);
        }
        return out;
    }
}
