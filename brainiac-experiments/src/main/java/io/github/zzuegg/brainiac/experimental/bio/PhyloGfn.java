package io.github.zzuegg.brainiac.experimental.bio;

import io.github.zzuegg.brainiac.experimental.data.BioSequences.Tree;
import io.github.zzuegg.brainiac.experimental.train.GFlowNet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * PhyloGFN: a generative flow network that builds rooted phylogenies bottom-up by merging two
 * subtrees of the current forest at each step, trained by trajectory balance (the unchanged
 * {@link GFlowNet}) to sample trees with probability proportional to
 * {@code exp(-parsimony / T)}. Unlike a single best-tree search it returns a diverse set of
 * near-optimal topologies.
 *
 * <p>A state is the forest, encoded canonically as the sorted bit masks of its internal
 * clusters; its parents are the forests obtained by splitting one of its non-trivial roots, so
 * the backward policy is uniform over them. The policy sees which slots are roots and, for every
 * pair of roots, the Fitch cost of merging them (normalised by the number of sites): the
 * features PhyloGFN builds from Fitch state sets.
 *
 * <p>Zhou et al., "PhyloGFN: Phylogenetic inference with generative flow networks" (ICLR 2024),
 * <a href="https://arxiv.org/abs/2310.08774">arXiv:2310.08774</a>.
 */
public final class PhyloGfn implements GFlowNet.Problem {

    private final int[][] alignment;
    private final int taxa;
    private final int sites;
    private final int[][] pairs;
    private final double temperature;
    private final double offset;
    private final Map<String, int[]> fitchCache = new HashMap<>();
    private final Map<String, Integer> costCache = new HashMap<>();

    /**
     * Creates the problem.
     *
     * @param alignment {@code [taxon][site]} DNA (at most 16 taxa)
     * @param temperature the reward temperature {@code T}
     * @param offset a parsimony score subtracted before scaling (e.g. a heuristic tree's score),
     *     so log rewards stay near zero
     */
    public PhyloGfn(int[][] alignment, double temperature, double offset) {
        if (alignment.length > 16) {
            throw new IllegalArgumentException("at most 16 taxa");
        }
        this.alignment = alignment;
        this.taxa = alignment.length;
        this.sites = alignment[0].length;
        this.temperature = temperature;
        this.offset = offset;
        List<int[]> p = new ArrayList<>();
        for (int i = 0; i < taxa; i++) {
            for (int j = i + 1; j < taxa; j++) {
                p.add(new int[] {i, j});
            }
        }
        this.pairs = p.toArray(new int[0][]);
    }

    private static int lowest(int mask) {
        return Integer.numberOfTrailingZeros(mask);
    }

    /** The root cluster of each slot (slot = lowest leaf of the cluster), 0 if not a root. */
    private int[] roots(int[] state) {
        int[] root = new int[taxa];
        for (int i = 0; i < taxa; i++) {
            root[i] = 1 << i;
        }
        int[] covered = new int[taxa];
        for (int c : state) {
            for (int i = 0; i < taxa; i++) {
                if ((c >> i & 1) != 0) {
                    covered[i] = Math.max(covered[i], Integer.bitCount(c));
                }
            }
        }
        // a cluster is a root if no larger cluster contains it
        Arrays.fill(root, 0);
        for (int i = 0; i < taxa; i++) {
            if (covered[i] == 0) {
                root[i] = 1 << i;
            }
        }
        for (int c : state) {
            boolean maximal = true;
            for (int d : state) {
                if (d != c && (d & c) == c) {
                    maximal = false;
                    break;
                }
            }
            if (maximal) {
                root[lowest(c)] = c;
            }
        }
        return root;
    }

    @Override
    public int[] initial() {
        return new int[0];
    }

    @Override
    public int actions() {
        return pairs.length + 1;
    }

    @Override
    public int stopAction() {
        return pairs.length;
    }

    @Override
    public boolean[] legal(int[] state) {
        boolean[] legal = new boolean[actions()];
        if (state.length == taxa - 1) {
            legal[stopAction()] = true;
            return legal;
        }
        int[] root = roots(state);
        for (int a = 0; a < pairs.length; a++) {
            legal[a] = root[pairs[a][0]] != 0 && root[pairs[a][1]] != 0;
        }
        return legal;
    }

    @Override
    public int[] step(int[] state, int action) {
        int[] root = roots(state);
        int merged = root[pairs[action][0]] | root[pairs[action][1]];
        int[] next = Arrays.copyOf(state, state.length + 1);
        next[state.length] = merged;
        Arrays.sort(next);
        return next;
    }

    @Override
    public int parents(int[] state) {
        int count = 0;
        for (int r : roots(state)) {
            count += Integer.bitCount(r) > 1 ? 1 : 0;
        }
        return count;
    }

    /** A subtree is identified by its cluster and every sub-cluster of it in the state. */
    private static String key(int cluster, int[] state) {
        StringBuilder b = new StringBuilder().append(cluster);
        for (int c : state) {
            if ((c & cluster) == c && c != cluster) {
                b.append(',').append(c);
            }
        }
        return b.toString();
    }

    private int[] fitchSets(int cluster, int[] state) {
        int[] cached = fitchCache.get(key(cluster, state));
        if (cached != null) {
            return cached;
        }
        throw new IllegalStateException("cluster not built");
    }

    /** Fitch sets of every cluster of a state, filling the cache; returns the state's total cost. */
    private int build(int[] state) {
        for (int i = 0; i < taxa; i++) {
            String leaf = key(1 << i, state);
            if (!fitchCache.containsKey(leaf)) {
                int[] s = new int[sites];
                for (int k = 0; k < sites; k++) {
                    s[k] = 1 << alignment[i][k];
                }
                fitchCache.put(leaf, s);
                costCache.put(leaf, 0);
            }
        }
        int[] sorted = state.clone();
        Arrays.sort(sorted);
        // clusters in order of size so children come first
        Integer[] bySize = Arrays.stream(sorted).boxed().toArray(Integer[]::new);
        Arrays.sort(bySize, (x, y) -> Integer.compare(Integer.bitCount(x), Integer.bitCount(y)));
        for (int idx = 0; idx < bySize.length; idx++) {
            int c = bySize[idx];
            String ck = key(c, sorted);
            if (fitchCache.containsKey(ck)) {
                continue;
            }
            // children: the two largest proper sub-clusters (or leaves) that partition c
            int[] parts = children(c, bySize, idx);
            int[] a = fitchSets(parts[0], sorted);
            int[] b = fitchSets(parts[1], sorted);
            int[] s = new int[sites];
            int cost = costCache.get(key(parts[0], sorted)) + costCache.get(key(parts[1], sorted));
            for (int k = 0; k < sites; k++) {
                int inter = a[k] & b[k];
                if (inter != 0) {
                    s[k] = inter;
                } else {
                    s[k] = a[k] | b[k];
                    cost++;
                }
            }
            fitchCache.put(ck, s);
            costCache.put(ck, cost);
        }
        int total = 0;
        for (int r : roots(state)) {
            if (r != 0) {
                total += costCache.get(key(r, sorted));
            }
        }
        return total;
    }

    private int[] children(int c, Integer[] bySize, int idx) {
        int first = 0;
        for (int j = idx - 1; j >= 0; j--) {
            int d = bySize[j];
            if ((d & c) == d && d != c) {
                first = d;
                break;
            }
        }
        if (first == 0) {
            first = 1 << lowest(c);
        }
        int rest = c & ~first;
        return new int[] {first, rest};
    }

    private int mergeCost(int a, int b, int[] state) {
        int[] x = fitchSets(a, state);
        int[] y = fitchSets(b, state);
        int cost = 0;
        for (int k = 0; k < sites; k++) {
            cost += (x[k] & y[k]) == 0 ? 1 : 0;
        }
        return cost;
    }

    @Override
    public float[] features(int[] state) {
        build(state);
        int[] sorted = state.clone();
        Arrays.sort(sorted);
        int[] root = roots(state);
        float[] f = new float[taxa + pairs.length];
        for (int i = 0; i < taxa; i++) {
            f[i] = root[i] != 0 ? 1 : 0;
        }
        for (int a = 0; a < pairs.length; a++) {
            int x = root[pairs[a][0]];
            int y = root[pairs[a][1]];
            f[taxa + a] = x != 0 && y != 0 ? (float) (mergeCost(x, y, sorted) * 10.0 / sites) : 0;
        }
        return f;
    }

    /**
     * The parsimony score of a complete state.
     *
     * @param state a terminal state
     * @return the Fitch parsimony score
     */
    public int parsimony(int[] state) {
        return build(state);
    }

    @Override
    public double logReward(int[] terminal) {
        return -(parsimony(terminal) - offset) / temperature;
    }

    /**
     * Converts a terminal state to a tree (unit branch lengths).
     *
     * @param state a terminal state
     * @return the tree
     */
    public Tree toTree(int[] state) {
        int nodes = 2 * taxa - 1;
        int[] parent = new int[nodes];
        Integer[] bySize = Arrays.stream(state).boxed().toArray(Integer[]::new);
        Arrays.sort(bySize, (x, y) -> Integer.compare(Integer.bitCount(x), Integer.bitCount(y)));
        Map<Integer, Integer> node = new HashMap<>();
        for (int i = 0; i < taxa; i++) {
            node.put(1 << i, i);
        }
        for (int idx = 0; idx < bySize.length; idx++) {
            node.put(bySize[idx], taxa + idx);
        }
        for (int idx = 0; idx < bySize.length; idx++) {
            int[] parts = children(bySize[idx], bySize, idx);
            parent[node.get(parts[0])] = taxa + idx;
            parent[node.get(parts[1])] = taxa + idx;
        }
        parent[nodes - 1] = -1;
        double[] len = new double[nodes];
        Arrays.fill(len, 0.1);
        return new Tree(parent, len, taxa);
    }

    /**
     * Trains a PhyloGFN sampler and draws trees from it.
     *
     * @param steps training steps
     * @param batch trajectories per step
     * @param samples trees drawn after training
     * @param seed the seed
     * @return the drawn terminal states
     */
    public List<int[]> trainAndSample(int steps, int batch, int samples, long seed) {
        GFlowNet gfn = GFlowNet.of(this, 64, 3e-3f, seed).exploration(0.05);
        SplittableRandom rnd = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            gfn.trainStep(batch, rnd);
        }
        List<int[]> out = new ArrayList<>();
        for (GFlowNet.Trajectory t : gfn.sample(samples, 0, rnd)) {
            out.add(t.terminal());
        }
        return out;
    }
}
