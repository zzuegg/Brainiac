package io.github.zzuegg.brainiac.experimental.circuits;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A random binary region tree over variables (the region graph of RAT-SPNs, Peharz et al., UAI
 * 2019, <a href="https://arxiv.org/abs/1806.01910">arXiv:1806.01910</a>, with one partition per
 * region): every region is split into two random halves until single variables remain. Nodes are
 * stored in post-order, so children always precede their parent and the root is last; a
 * structured-decomposable circuit built on one tree can be squared in polynomial time.
 */
public final class RegionTree {

    private final int[] left;
    private final int[] right;
    private final int[] variable;

    private RegionTree(int[] left, int[] right, int[] variable) {
        this.left = left;
        this.right = right;
        this.variable = variable;
    }

    /**
     * Draws a random tree.
     *
     * @param variables the number of variables
     * @param seed the seed
     * @return the tree
     */
    public static RegionTree random(int variables, long seed) {
        int[] all = new int[variables];
        Arrays.setAll(all, i -> i);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = variables - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = all[i];
            all[i] = all[j];
            all[j] = t;
        }
        List<int[]> nodes = new ArrayList<>();   // {left, right, variable}
        build(all, nodes);
        int n = nodes.size();
        int[] l = new int[n];
        int[] rr = new int[n];
        int[] v = new int[n];
        for (int k = 0; k < n; k++) {
            l[k] = nodes.get(k)[0];
            rr[k] = nodes.get(k)[1];
            v[k] = nodes.get(k)[2];
        }
        return new RegionTree(l, rr, v);
    }

    private static int build(int[] vars, List<int[]> nodes) {
        if (vars.length == 1) {
            nodes.add(new int[] {-1, -1, vars[0]});
            return nodes.size() - 1;
        }
        int half = vars.length / 2;
        int a = build(Arrays.copyOfRange(vars, 0, half), nodes);
        int b = build(Arrays.copyOfRange(vars, half, vars.length), nodes);
        nodes.add(new int[] {a, b, -1});
        return nodes.size() - 1;
    }

    /** @return the number of nodes */
    public int size() {
        return left.length;
    }

    /** @return the root (the last node) */
    public int root() {
        return left.length - 1;
    }

    /**
     * @param node a node
     * @return whether it is a leaf (a single variable)
     */
    public boolean isLeaf(int node) {
        return variable[node] >= 0;
    }

    /**
     * @param node a leaf
     * @return its variable
     */
    public int variable(int node) {
        return variable[node];
    }

    /**
     * @param node an inner node
     * @return its left child
     */
    public int left(int node) {
        return left[node];
    }

    /**
     * @param node an inner node
     * @return its right child
     */
    public int right(int node) {
        return right[node];
    }
}
