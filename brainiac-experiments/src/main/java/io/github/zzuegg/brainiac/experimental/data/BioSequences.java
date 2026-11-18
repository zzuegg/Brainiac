package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Synthetic biological sequences with known ground truth, encoded as integer residues:
 * homologous pairs with their true alignment, alignments evolved along random trees under
 * Jukes-Cantor or Kimura two-parameter substitution, sequences with a planted motif given as a
 * position weight matrix, families sampled from profile models, Potts-model samples with a
 * planted contact map (by Gibbs sampling), and RNA sequences with planted nested stems.
 *
 * <p>DNA uses {@code ACGT} (0..3), RNA {@code ACGU} (0..3) and proteins the 20 amino acids in
 * the order {@link #PROTEIN}. Protein substitutions follow a hidden block structure (residues
 * in the same physico-chemical group swap more often), so a learned substitution matrix has
 * something to recover beyond identity.
 */
public final class BioSequences {

    /** The DNA alphabet. */
    public static final String DNA = "ACGT";
    /** The RNA alphabet. */
    public static final String RNA = "ACGU";
    /** The protein alphabet. */
    public static final String PROTEIN = "ACDEFGHIKLMNPQRSTVWY";
    /** The hidden substitution groups of {@link #PROTEIN} (index of the group of each residue). */
    private static final int[] PROTEIN_GROUP = {0, 1, 2, 2, 3, 0, 4, 4, 4, 5, 5, 2, 0, 2, 4, 1, 1, 5, 3, 3};

    private BioSequences() {
    }

    /**
     * The hidden substitution group of a protein residue (residues in a group swap more often).
     *
     * @param residue an index into {@link #PROTEIN}
     * @return the group index
     */
    public static int proteinGroup(int residue) {
        return PROTEIN_GROUP[residue];
    }

    /**
     * Turns residues into a string.
     *
     * @param seq residues
     * @param alphabet the alphabet
     * @return the letters
     */
    public static String text(int[] seq, String alphabet) {
        StringBuilder b = new StringBuilder(seq.length);
        for (int s : seq) {
            b.append(alphabet.charAt(s));
        }
        return b.toString();
    }

    /**
     * A homologous pair with its true alignment.
     *
     * @param a the first sequence
     * @param b the second sequence
     * @param match for each position of {@code a}, the aligned position in {@code b}, or -1
     */
    public record Pair(int[] a, int[] b, int[] match) {
    }

    /**
     * Evolves {@code b} from {@code a}: each site is substituted with probability
     * {@code divergence} (protein: 70% within its hidden group), and indels of geometric length
     * (mean 2) start at each site with probability {@code indel}.
     *
     * @param length the length of {@code a}
     * @param divergence the substitution probability per site
     * @param indel the indel start probability per site
     * @param alphabetSize 4 or 20
     * @param rnd the random source
     * @return the pair
     */
    public static Pair homologousPair(int length, double divergence, double indel, int alphabetSize,
            SplittableRandom rnd) {
        int[] a = randomSequence(length, alphabetSize, rnd);
        List<Integer> b = new ArrayList<>();
        int[] match = new int[length];
        Arrays.fill(match, -1);
        int i = 0;
        while (i < length) {
            if (rnd.nextDouble() < indel / 2) {
                int len = geometric(rnd);
                for (int t = 0; t < len; t++) {
                    b.add(rnd.nextInt(alphabetSize));
                }
            }
            if (rnd.nextDouble() < indel / 2) {
                i += geometric(rnd);
                continue;
            }
            match[i] = b.size();
            b.add(rnd.nextDouble() < divergence ? substitute(a[i], alphabetSize, rnd) : a[i]);
            i++;
        }
        return new Pair(a, b.stream().mapToInt(Integer::intValue).toArray(), match);
    }

    private static int geometric(SplittableRandom rnd) {
        int len = 1;
        while (rnd.nextDouble() < 0.5) {
            len++;
        }
        return len;
    }

    private static int substitute(int r, int alphabetSize, SplittableRandom rnd) {
        if (alphabetSize == 20 && rnd.nextDouble() < 0.7) {
            int g = PROTEIN_GROUP[r];
            int[] same = java.util.stream.IntStream.range(0, 20).filter(x -> x != r && PROTEIN_GROUP[x] == g)
                    .toArray();
            if (same.length > 0) {
                return same[rnd.nextInt(same.length)];
            }
        }
        int s = rnd.nextInt(alphabetSize - 1);
        return s >= r ? s + 1 : s;
    }

    /**
     * A uniformly random sequence.
     *
     * @param length the length
     * @param alphabetSize the alphabet size
     * @param rnd the random source
     * @return the residues
     */
    public static int[] randomSequence(int length, int alphabetSize, SplittableRandom rnd) {
        int[] s = new int[length];
        for (int i = 0; i < length; i++) {
            s[i] = rnd.nextInt(alphabetSize);
        }
        return s;
    }

    /**
     * A rooted binary tree over {@code taxa} leaves: nodes {@code 0..taxa-1} are leaves, the
     * root is the last node, {@code parent[root] = -1}.
     *
     * @param parent the parent of each node
     * @param length the branch length (expected substitutions per site) above each node
     * @param taxa the number of leaves
     */
    public record Tree(int[] parent, double[] length, int taxa) {

        /** @return the root index */
        public int root() {
            return parent.length - 1;
        }

        /**
         * The non-trivial bipartitions (splits) of the leaves, each as a canonical bit string
         * (the side not containing leaf 0), for Robinson-Foulds distances.
         *
         * @return the set of splits
         */
        public java.util.Set<String> splits() {
            int nodes = parent.length;
            boolean[][] below = new boolean[nodes][taxa];
            for (int leaf = 0; leaf < taxa; leaf++) {
                for (int v = leaf; v >= 0; v = parent[v]) {
                    below[v][leaf] = true;
                }
            }
            java.util.Set<String> out = new java.util.HashSet<>();
            for (int v = taxa; v < nodes; v++) {
                int count = 0;
                for (boolean x : below[v]) {
                    count += x ? 1 : 0;
                }
                if (count < 2 || count > taxa - 2) {
                    continue;
                }
                StringBuilder sb = new StringBuilder();
                for (int l = 0; l < taxa; l++) {
                    sb.append(below[v][l] != below[v][0] ? '1' : '0');
                }
                out.add(sb.toString());
            }
            return out;
        }

        /**
         * The normalised Robinson-Foulds distance.
         *
         * @param other another tree on the same leaves
         * @return symmetric difference of the split sets over their total size, in [0, 1]
         */
        public double robinsonFoulds(Tree other) {
            java.util.Set<String> a = splits();
            java.util.Set<String> b = other.splits();
            int common = 0;
            for (String s : a) {
                common += b.contains(s) ? 1 : 0;
            }
            int total = a.size() + b.size();
            return total == 0 ? 0 : (total - 2.0 * common) / total;
        }
    }

    /**
     * A random tree by coalescence: lineages merge in random order; with a clock all leaves are
     * equidistant from the root, otherwise branch lengths are scaled by a log-normal rate with
     * spread {@code 1.2} per branch (strongly non-clock).
     *
     * @param taxa leaves
     * @param depth the expected root-to-leaf distance in substitutions per site
     * @param clock whether a molecular clock holds
     * @param rnd the random source
     * @return the tree
     */
    public static Tree randomTree(int taxa, double depth, boolean clock, SplittableRandom rnd) {
        int nodes = 2 * taxa - 1;
        int[] parent = new int[nodes];
        double[] height = new double[nodes];
        List<Integer> live = new ArrayList<>();
        for (int i = 0; i < taxa; i++) {
            live.add(i);
        }
        double h = 0;
        for (int v = taxa; v < nodes; v++) {
            int n = live.size();
            h += -Math.log(1 - rnd.nextDouble()) * 2.0 / (n * (n - 1.0));
            int a = live.remove(rnd.nextInt(live.size()));
            int b = live.remove(rnd.nextInt(live.size()));
            parent[a] = v;
            parent[b] = v;
            height[v] = h;
            live.add(v);
        }
        parent[nodes - 1] = -1;
        double scale = depth / height[nodes - 1];
        double[] len = new double[nodes];
        for (int v = 0; v < nodes - 1; v++) {
            len[v] = (height[parent[v]] - height[v]) * scale;
            if (!clock) {
                len[v] *= Math.exp(1.2 * ChannelCodes.gaussian(rnd) - 0.72);
            }
            len[v] = Math.max(len[v], 1e-3);
        }
        return new Tree(parent, len, taxa);
    }

    /**
     * Evolves an alignment of DNA along a tree: Jukes-Cantor when {@code kappa = 1}, Kimura's
     * two-parameter model with transition/transversion ratio {@code kappa} otherwise.
     *
     * @param tree the tree
     * @param sites alignment columns
     * @param kappa the transition/transversion rate ratio
     * @param rnd the random source
     * @return {@code [taxon][site]}, leaves only
     */
    public static int[][] evolve(Tree tree, int sites, double kappa, SplittableRandom rnd) {
        int nodes = tree.parent().length;
        int[][] seq = new int[nodes][];
        seq[tree.root()] = randomSequence(sites, 4, rnd);
        List<List<Integer>> children = childLists(tree);
        java.util.ArrayDeque<Integer> stack = new java.util.ArrayDeque<>();
        stack.push(tree.root());
        while (!stack.isEmpty()) {
            int v = stack.pop();
            for (int c : children.get(v)) {
                double[][] p = k2pMatrix(tree.length()[c], kappa);
                seq[c] = new int[sites];
                for (int s = 0; s < sites; s++) {
                    double u = rnd.nextDouble();
                    int from = seq[v][s];
                    int to = 3;
                    double acc = 0;
                    for (int x = 0; x < 4; x++) {
                        acc += p[from][x];
                        if (u < acc) {
                            to = x;
                            break;
                        }
                    }
                    seq[c][s] = to;
                }
                stack.push(c);
            }
        }
        return Arrays.copyOf(seq, tree.taxa());
    }

    /**
     * The children of each node.
     *
     * @param tree the tree
     * @return child lists
     */
    public static List<List<Integer>> childLists(Tree tree) {
        List<List<Integer>> children = new ArrayList<>();
        for (int v = 0; v < tree.parent().length; v++) {
            children.add(new ArrayList<>());
        }
        for (int v = 0; v < tree.parent().length; v++) {
            if (tree.parent()[v] >= 0) {
                children.get(tree.parent()[v]).add(v);
            }
        }
        return children;
    }

    /**
     * The Kimura two-parameter transition matrix (Jukes-Cantor at {@code kappa = 1}) for a
     * branch of {@code t} expected substitutions per site; purines A, G are 0, 2.
     *
     * @param t the branch length
     * @param kappa the transition/transversion ratio
     * @return {@code P[from][to]}
     */
    public static double[][] k2pMatrix(double t, double kappa) {
        double beta = t / (kappa + 2);
        double alpha = kappa * beta;
        double pTransversion = 0.25 * (1 - Math.exp(-4 * beta));
        double pTransition = 0.25 + 0.25 * Math.exp(-4 * beta) - 0.5 * Math.exp(-2 * (alpha + beta));
        double[][] p = new double[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i == j) {
                    continue;
                }
                p[i][j] = (i % 2 == j % 2) ? pTransition : pTransversion;
            }
            p[i][i] = 1 - pTransition - 2 * pTransversion;
        }
        return p;
    }

    /**
     * Sequences with a planted motif.
     *
     * @param sequences {@code [i]} the DNA residues
     * @param bound whether sequence i is a positive
     * @param sites the motif start in each sequence, or -1
     * @param pwm the true position weight matrix {@code [width][4]}
     */
    public record MotifSet(int[][] sequences, boolean[] bound, int[] sites, double[][] pwm) {
    }

    /**
     * Half positives, half negatives, with background {@code (0.3, 0.2, 0.2, 0.3)}; each positive
     * carries one motif instance with probability {@code occupancy}. The PWM's columns are
     * Dirichlet-like draws peaked on a random consensus (information about 1.4 bits per column).
     *
     * @param n sequences
     * @param length sequence length
     * @param width motif width
     * @param occupancy the share of positives with a site
     * @param rnd the random source
     * @return the set
     */
    public static MotifSet motifs(int n, int length, int width, double occupancy, SplittableRandom rnd) {
        double[][] pwm = new double[width][4];
        for (int w = 0; w < width; w++) {
            int consensus = rnd.nextInt(4);
            double peak = 0.6 + 0.35 * rnd.nextDouble();
            for (int x = 0; x < 4; x++) {
                pwm[w][x] = x == consensus ? peak : (1 - peak) / 3;
            }
        }
        double[] background = {0.3, 0.2, 0.2, 0.3};
        int[][] seqs = new int[n][length];
        boolean[] bound = new boolean[n];
        int[] sites = new int[n];
        for (int i = 0; i < n; i++) {
            for (int p = 0; p < length; p++) {
                seqs[i][p] = draw(background, rnd);
            }
            bound[i] = i % 2 == 0;
            sites[i] = -1;
            if (bound[i] && rnd.nextDouble() < occupancy) {
                int s = rnd.nextInt(length - width + 1);
                sites[i] = s;
                for (int w = 0; w < width; w++) {
                    seqs[i][s + w] = draw(pwm[w], rnd);
                }
            }
        }
        return new MotifSet(seqs, bound, sites, pwm);
    }

    /**
     * Draws from a categorical distribution.
     *
     * @param p probabilities
     * @param rnd the random source
     * @return the index
     */
    public static int draw(double[] p, SplittableRandom rnd) {
        double u = rnd.nextDouble();
        double acc = 0;
        for (int i = 0; i < p.length; i++) {
            acc += p[i];
            if (u < acc) {
                return i;
            }
        }
        return p.length - 1;
    }

    /**
     * A protein family sampled from a profile: each column has a consensus residue kept with
     * probability {@code identity} (otherwise a within-group or random substitute), with
     * insertions and deletions at rate {@code indel} per column.
     *
     * @param members family members
     * @param length the profile length
     * @param identity the conservation of each column
     * @param indel the indel rate per column
     * @param rnd the random source
     * @return the members
     */
    public static List<int[]> family(int members, int length, double identity, double indel, SplittableRandom rnd) {
        int[] consensus = randomSequence(length, 20, rnd);
        double[] conservation = new double[length];
        for (int c = 0; c < length; c++) {
            // a few highly conserved positions, the rest at the target identity
            conservation[c] = rnd.nextDouble() < 0.25 ? Math.min(0.95, identity + 0.4) : identity;
        }
        List<int[]> out = new ArrayList<>();
        for (int m = 0; m < members; m++) {
            List<Integer> s = new ArrayList<>();
            for (int c = 0; c < length; c++) {
                if (rnd.nextDouble() < indel) {
                    s.add(rnd.nextInt(20));
                }
                if (rnd.nextDouble() < indel) {
                    continue;
                }
                s.add(rnd.nextDouble() < conservation[c] ? consensus[c] : substitute(consensus[c], 20, rnd));
            }
            out.add(s.stream().mapToInt(Integer::intValue).toArray());
        }
        return out;
    }

    /**
     * Potts-model samples with a planted contact map.
     *
     * @param sequences {@code [sample][position]}
     * @param contacts the planted contacts (symmetric)
     */
    public record PottsSamples(int[][] sequences, boolean[][] contacts) {
    }

    /**
     * Samples a Potts model {@code p(s) ~ exp(sum h_i(s_i) + sum J_ij(s_i, s_j))} by Gibbs
     * sampling. Contacts: a chain of pairs {@code (i, i+3)} for {@code i = 0, 3, 6, ...} (local,
     * and chained so that transitive correlations such as {@code (0, 6)} confound mutual
     * information) plus {@code L/3} random long-range pairs ({@code |i-j| >= 6}); each coupling rewards a
     * random permutation of matching states with {@code strength} and penalises the rest
     * mildly. With {@code phylogeny} the samples come in clusters of 8 descendants of a shared
     * ancestor sample with only 2 sweeps of divergence, a phylogenetic-correlation stress test.
     *
     * @param length positions L
     * @param states states q
     * @param n samples
     * @param strength the coupling strength
     * @param phylogeny whether to add clustered phylogenetic correlation
     * @param rnd the random source
     * @return the samples and contacts
     */
    public static PottsSamples potts(int length, int states, int n, double strength, boolean phylogeny,
            SplittableRandom rnd) {
        boolean[][] contact = new boolean[length][length];
        for (int i = 0; i + 3 < length; i += 3) {
            contact[i][i + 3] = true;
            contact[i + 3][i] = true;
        }
        int longRange = length / 3;
        while (longRange > 0) {
            int i = rnd.nextInt(length);
            int j = rnd.nextInt(length);
            if (Math.abs(i - j) >= 6 && !contact[i][j]) {
                contact[i][j] = true;
                contact[j][i] = true;
                longRange--;
            }
        }
        double[][] h = new double[length][states];
        for (double[] row : h) {
            for (int a = 0; a < states; a++) {
                row[a] = 0.3 * ChannelCodes.gaussian(rnd);
            }
        }
        double[][][][] j = new double[length][length][][];
        for (int a = 0; a < length; a++) {
            for (int b = a + 1; b < length; b++) {
                if (!contact[a][b]) {
                    continue;
                }
                int[] perm = permutation(states, rnd);
                double[][] m = new double[states][states];
                for (int x = 0; x < states; x++) {
                    for (int y = 0; y < states; y++) {
                        m[x][y] = perm[x] == y ? strength : -strength / (states - 1);
                    }
                }
                j[a][b] = m;
                double[][] t = new double[states][states];
                for (int x = 0; x < states; x++) {
                    for (int y = 0; y < states; y++) {
                        t[y][x] = m[x][y];
                    }
                }
                j[b][a] = t;
            }
        }
        int[] s = randomSequence(length, states, rnd);
        for (int sweep = 0; sweep < 200; sweep++) {
            gibbsSweep(s, h, j, states, rnd);
        }
        int[][] out = new int[n][];
        int[] ancestor = s.clone();
        for (int i = 0; i < n; i++) {
            if (phylogeny) {
                if (i % 8 == 0) {
                    for (int sweep = 0; sweep < 20; sweep++) {
                        gibbsSweep(ancestor, h, j, states, rnd);
                    }
                }
                int[] c = ancestor.clone();
                for (int sweep = 0; sweep < 2; sweep++) {
                    gibbsSweep(c, h, j, states, rnd);
                }
                out[i] = c;
            } else {
                for (int sweep = 0; sweep < 5; sweep++) {
                    gibbsSweep(s, h, j, states, rnd);
                }
                out[i] = s.clone();
            }
        }
        return new PottsSamples(out, contact);
    }

    private static int[] permutation(int n, SplittableRandom rnd) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int k = rnd.nextInt(i + 1);
            int t = p[i];
            p[i] = p[k];
            p[k] = t;
        }
        return p;
    }

    private static void gibbsSweep(int[] s, double[][] h, double[][][][] j, int states, SplittableRandom rnd) {
        int length = s.length;
        double[] e = new double[states];
        for (int i = 0; i < length; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int a = 0; a < states; a++) {
                double v = h[i][a];
                for (int k = 0; k < length; k++) {
                    if (j[i][k] != null) {
                        v += j[i][k][a][s[k]];
                    }
                }
                e[a] = v;
                max = Math.max(max, v);
            }
            double z = 0;
            for (int a = 0; a < states; a++) {
                e[a] = Math.exp(e[a] - max);
                z += e[a];
            }
            for (int a = 0; a < states; a++) {
                e[a] /= z;
            }
            s[i] = draw(e, rnd);
        }
    }

    /**
     * An RNA sequence with its planted secondary structure.
     *
     * @param sequence residues in {@code ACGU}
     * @param pairs the partner of each position, or -1
     */
    public record RnaStructure(int[] sequence, int[] pairs) {
    }

    /**
     * An RNA sequence with planted nested stems: the interval is filled recursively with stems
     * of 4-7 base pairs (70% GC, 20% AU, 10% GU pairs) closing hairpins of 4-8 unpaired bases or
     * multiloops; unpaired bases are A-rich ({@code 0.4, 0.2, 0.2, 0.2}). The pairing energies
     * of this generator are not the energy model any folding method uses.
     *
     * @param length the length
     * @param rnd the random source
     * @return the structure
     */
    public static RnaStructure rna(int length, SplittableRandom rnd) {
        int[] seq = new int[length];
        int[] pairs = new int[length];
        Arrays.fill(pairs, -1);
        double[] loop = {0.4, 0.2, 0.2, 0.2};
        for (int i = 0; i < length; i++) {
            seq[i] = draw(loop, rnd);
        }
        fill(seq, pairs, 0, length - 1, rnd, 0);
        return new RnaStructure(seq, pairs);
    }

    private static void fill(int[] seq, int[] pairs, int lo, int hi, SplittableRandom rnd, int depth) {
        int span = hi - lo + 1;
        if (span < 14 || depth > 3) {
            return;
        }
        // place one or two stems side by side in this interval
        int stems = span > 40 && rnd.nextDouble() < 0.6 ? 2 : 1;
        int start = lo + rnd.nextInt(3);
        int room = (hi - start + 1) / stems;
        for (int st = 0; st < stems; st++) {
            int a = start + st * room;
            int b = Math.min(hi, a + room - 1 - rnd.nextInt(3));
            int len = 4 + rnd.nextInt(4);
            if (b - a + 1 < 2 * len + 4) {
                continue;
            }
            for (int k = 0; k < len; k++) {
                int i = a + k;
                int j = b - k;
                double u = rnd.nextDouble();
                int x;
                int y;
                if (u < 0.7) {
                    x = rnd.nextBoolean() ? 1 : 2;
                    y = 3 - x;
                } else if (u < 0.9) {
                    x = rnd.nextBoolean() ? 0 : 3;
                    y = 3 - x;
                } else {
                    x = rnd.nextBoolean() ? 2 : 3;
                    y = x == 2 ? 3 : 2;
                }
                seq[i] = x;
                seq[j] = y;
                pairs[i] = j;
                pairs[j] = i;
            }
            int innerLo = a + len;
            int innerHi = b - len;
            if (innerHi - innerLo + 1 > 8) {
                fill(seq, pairs, innerLo + 1, innerHi - 1, rnd, depth + 1);
            }
        }
    }

    /**
     * Whether two RNA bases can pair canonically (Watson-Crick or GU wobble).
     *
     * @param x a base in {@code ACGU}
     * @param y a base
     * @return whether they pair
     */
    public static boolean canPair(int x, int y) {
        return x + y == 3 || (x == 2 && y == 3) || (x == 3 && y == 2);
    }
}
