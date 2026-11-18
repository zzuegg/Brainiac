package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;

import java.util.ArrayList;
import java.util.List;

/**
 * A routing instance as a constructive decision process, shared by the neural
 * combinatorial-optimisation solvers: the travelling salesman problem (visit every node
 * once, return to the first) and the capacitated vehicle routing problem (node 0 is the
 * depot; visit every customer once, return to the depot whenever the next customer's demand
 * would not fit).
 *
 * <p>A {@link State} tracks the partial solution, gives the mask of legal next nodes and the
 * length so far. Node features are {@code (x - 1/2, y - 1/2)} (centred, so no large common component saturates the clipped logits) for the TSP and
 * the same plus {@code (demand / capacity, isDepot)} for the CVRP. {@link #symmetry} gives POMO's
 * eight coordinate symmetries of the unit square (they leave every tour length unchanged).
 *
 * <p>Papers: Kool, van Hoof and Welling, "Attention, Learn to Solve Routing Problems!" (ICLR
 * 2019), <a href="https://arxiv.org/abs/1803.08475">arXiv:1803.08475</a>; Kwon et al.,
 * "POMO" (NeurIPS 2020), <a href="https://arxiv.org/abs/2010.16011">arXiv:2010.16011</a>.
 */
public final class RoutingEnv {

    /** The problem kind. */
    public enum Kind {
        /** travelling salesman */
        TSP,
        /** capacitated vehicle routing */
        CVRP
    }

    private final Kind kind;
    private final double[][] coords;
    private final double[][] dist;
    private final double[] demand;
    private final double capacity;

    private RoutingEnv(Kind kind, double[][] coords, double[] demand, double capacity) {
        this.kind = kind;
        this.coords = coords;
        this.dist = ClassicHeuristics.distances(coords);
        this.demand = demand;
        this.capacity = capacity;
    }

    /**
     * A TSP instance.
     *
     * @param coords {@code [n][2]}
     * @return the environment
     */
    public static RoutingEnv tsp(double[][] coords) {
        return new RoutingEnv(Kind.TSP, coords, null, 0);
    }

    /**
     * A CVRP instance.
     *
     * @param coords {@code [n][2]}, depot first
     * @param demand demands, {@code demand[0] = 0}
     * @param capacity the capacity
     * @return the environment
     */
    public static RoutingEnv cvrp(double[][] coords, double[] demand, double capacity) {
        return new RoutingEnv(Kind.CVRP, coords, demand, capacity);
    }

    /**
     * A CVRP environment for a generated instance.
     *
     * @param c the instance
     * @return the environment
     */
    public static RoutingEnv cvrp(CombinatorialInstances.Cvrp c) {
        return cvrp(c.coords(), c.demand(), c.capacity());
    }

    /**
     * @param kind the kind
     * @return node features per node
     */
    public static int features(Kind kind) {
        return kind == Kind.TSP ? 2 : 4;
    }

    /** @return the kind */
    public Kind kind() {
        return kind;
    }

    /** @return nodes (customers plus the depot for CVRP) */
    public int nodes() {
        return coords.length;
    }

    /** @return the coordinates */
    public double[][] coords() {
        return coords;
    }

    /** @return the distance matrix */
    public double[][] distances() {
        return dist;
    }

    /** @return the demands, or {@code null} for the TSP */
    public double[] demand() {
        return demand;
    }

    /** @return the capacity (0 for the TSP) */
    public double capacity() {
        return capacity;
    }

    /** @return the node features, flattened {@code [nodes * features]} */
    public float[] nodeFeatures() {
        int f = features(kind);
        float[] out = new float[coords.length * f];
        for (int i = 0; i < coords.length; i++) {
            out[i * f] = (float) (coords[i][0] - 0.5);
            out[i * f + 1] = (float) (coords[i][1] - 0.5);
            if (kind == Kind.CVRP) {
                out[i * f + 2] = (float) (demand[i] / capacity);
                out[i * f + 3] = i == 0 ? 1 : 0;
            }
        }
        return out;
    }

    /**
     * The length of a finished solution.
     *
     * @param solution a tour (TSP) or a sequence with depot returns (CVRP)
     * @return its length
     */
    public double length(int[] solution) {
        return kind == Kind.TSP ? ClassicHeuristics.tourLength(dist, solution)
                : ClassicHeuristics.routeLength(dist, solution);
    }

    /**
     * One of the eight symmetries of the unit square: {@code k} in 0..7 picks
     * {@code (x,y), (y,x), (1-x,y), (y,1-x), (x,1-y), (1-y,x), (1-x,1-y), (1-y,1-x)}.
     *
     * @param k the symmetry
     * @return the transformed instance (same solution lengths)
     */
    public RoutingEnv symmetry(int k) {
        double[][] c = new double[coords.length][2];
        for (int i = 0; i < coords.length; i++) {
            double x = coords[i][0];
            double y = coords[i][1];
            double[] p = switch (k & 7) {
                case 0 -> new double[] {x, y};
                case 1 -> new double[] {y, x};
                case 2 -> new double[] {1 - x, y};
                case 3 -> new double[] {y, 1 - x};
                case 4 -> new double[] {x, 1 - y};
                case 5 -> new double[] {1 - y, x};
                case 6 -> new double[] {1 - x, 1 - y};
                default -> new double[] {1 - y, 1 - x};
            };
            c[i] = p;
        }
        return new RoutingEnv(kind, c, demand, capacity);
    }

    /**
     * Rotates the instance about the centre of the square (rescaled into it if needed).
     *
     * @param angle radians
     * @return the rotated instance; lengths scale by the returned instance's
     *     {@link #scaleFrom(RoutingEnv)}
     */
    public RoutingEnv rotated(double angle) {
        CombinatorialInstances.Tsp t = CombinatorialInstances.rotated(new CombinatorialInstances.Tsp(coords, 0, false),
                angle, false);
        return new RoutingEnv(kind, t.coords(), demand, capacity);
    }

    /**
     * The ratio of this instance's distances to another's for the same nodes (1 unless
     * rescaled by {@link #rotated}).
     *
     * @param original the instance this was made from
     * @return {@code d_this / d_original} for the first pair of distinct nodes
     */
    public double scaleFrom(RoutingEnv original) {
        return dist[0][1] / Math.max(1e-12, original.dist[0][1]);
    }

    /** @return a fresh state */
    public State start() {
        return new State();
    }

    /** A partial solution. */
    public final class State {
        private final boolean[] visited = new boolean[coords.length];
        private final List<Integer> sequence = new ArrayList<>();
        private int current = -1;
        private int first = -1;
        private int left;
        private double load;
        private double length;

        private State() {
            left = kind == Kind.TSP ? coords.length : coords.length - 1;
            if (kind == Kind.CVRP) {
                current = 0;
            }
        }

        /** @return whether every node (customer) has been visited */
        public boolean done() {
            return left == 0;
        }

        /** @return the current node, or -1 before the first TSP step */
        public int current() {
            return current;
        }

        /** @return the first node (TSP) or -1 */
        public int first() {
            return first;
        }

        /** @return the remaining capacity as a share of the capacity (CVRP), else 0 */
        public double remaining() {
            return kind == Kind.CVRP ? (capacity - load) / capacity : 0;
        }

        /**
         * The legal next nodes. When done, only a no-op node (node 0 for CVRP, the current
         * node for the TSP) is legal; it does not change the state.
         *
         * @return the mask
         */
        public boolean[] mask() {
            boolean[] m = new boolean[coords.length];
            if (done()) {
                m[kind == Kind.CVRP ? 0 : current] = true;
                return m;
            }
            boolean any = false;
            for (int i = 0; i < coords.length; i++) {
                if (kind == Kind.TSP) {
                    m[i] = !visited[i];
                } else if (i > 0) {
                    m[i] = !visited[i] && load + demand[i] <= capacity + 1e-9;
                    any |= m[i];
                }
            }
            if (kind == Kind.CVRP) {
                m[0] = current != 0 || !any;
            }
            return m;
        }

        /**
         * Moves to a node (a no-op when done).
         *
         * @param node a legal node
         */
        public void step(int node) {
            if (done()) {
                return;
            }
            if (!mask()[node]) {
                throw new IllegalArgumentException("node " + node + " is not legal here");
            }
            if (current >= 0) {
                length += dist[current][node];
            }
            if (kind == Kind.TSP) {
                if (first < 0) {
                    first = node;
                }
                visited[node] = true;
                left--;
            } else if (node == 0) {
                load = 0;
            } else {
                visited[node] = true;
                load += demand[node];
                left--;
            }
            sequence.add(node);
            current = node;
        }

        /** @return the length so far (closing edge excluded) */
        public double partialLength() {
            return length;
        }

        /** @return the solution as built so far */
        public int[] solution() {
            return sequence.stream().mapToInt(Integer::intValue).toArray();
        }
    }
}
