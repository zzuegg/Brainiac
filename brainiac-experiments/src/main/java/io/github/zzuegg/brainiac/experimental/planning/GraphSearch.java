package io.github.zzuegg.brainiac.experimental.planning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.ToDoubleFunction;

/**
 * Best-first graph search for planning: breadth-first search, Dijkstra, A* (Hart, Nilsson &amp;
 * Raphael 1968), weighted A* ({@code f = g + w h}, Pohl 1970, cost within a factor {@code w}),
 * greedy best-first search, and jump point search on uniform-cost 8-connected grids (Harabor
 * &amp; Grastien, AAAI 2011), which prunes symmetric paths by jumping along straight lines to
 * forced neighbours and keeps A*'s optimal cost with far fewer expansions.
 *
 * <p>Grids are row-major {@code boolean[] blocked}; 8-connected moves never cut a corner (a
 * diagonal step needs both orthogonal neighbours free), which is the rule this jump point
 * search's pruning matches. Ties on {@code f} break towards larger {@code g} (deeper nodes)
 * by default. A generic {@link #search} runs the same algorithms on any implicit graph with
 * unit edge costs, for puzzles and learned heuristics.
 */
public final class GraphSearch {

    private GraphSearch() {
    }

    /** The search algorithm. */
    public enum Method {
        /** Breadth-first search (unit costs; optimal for 4-connected grids). */
        BFS,
        /** Uniform-cost search. */
        DIJKSTRA,
        /** A*. */
        ASTAR,
        /** Weighted A*. */
        WEIGHTED_ASTAR,
        /** Greedy best-first search on {@code h} alone. */
        GREEDY,
        /** Jump point search (8-connected grids only). */
        JPS
    }

    /** A grid heuristic. */
    public enum Heuristic {
        /** No heuristic. */
        ZERO,
        /** {@code |dx| + |dy|}, admissible for 4-connected grids. */
        MANHATTAN,
        /** {@code max + (sqrt 2 - 1) min}, exact on empty 8-connected grids. */
        OCTILE,
        /** Straight-line distance. */
        EUCLIDEAN
    }

    /**
     * A search result.
     *
     * @param path the cells (or states) from start to goal, empty if none
     * @param cost the path cost, infinite if not found
     * @param expanded the number of nodes expanded (popped and not stale)
     * @param generated the number of nodes pushed
     */
    public record Result(List<Integer> path, double cost, int expanded, int generated) {

        /** @return whether a path was found */
        public boolean found() {
            return !path.isEmpty();
        }
    }

    /**
     * The value of a grid heuristic.
     *
     * @param h the heuristic
     * @param ax a column
     * @param ay a row
     * @param bx a column
     * @param by a row
     * @return the estimate
     */
    public static double heuristic(Heuristic h, int ax, int ay, int bx, int by) {
        int dx = Math.abs(ax - bx);
        int dy = Math.abs(ay - by);
        return switch (h) {
            case ZERO -> 0;
            case MANHATTAN -> dx + dy;
            case OCTILE -> Math.max(dx, dy) + (Math.sqrt(2) - 1) * Math.min(dx, dy);
            case EUCLIDEAN -> Math.hypot(dx, dy);
        };
    }

    private record Node(int cell, double f, double g) {
    }

    private static PriorityQueue<Node> queue() {
        return new PriorityQueue<>((a, b) -> {
            int c = Double.compare(a.f, b.f);
            return c != 0 ? c : Double.compare(b.g, a.g);
        });
    }

    /**
     * Searches a grid.
     *
     * @param blocked per cell whether it is an obstacle
     * @param width the width
     * @param height the height
     * @param start the start cell
     * @param goal the goal cell
     * @param method the algorithm
     * @param heuristic the heuristic (ignored by BFS and Dijkstra)
     * @param weight {@code w} for weighted A* (1 otherwise)
     * @param diagonal 8-connected moves (required for JPS)
     * @return the result
     */
    public static Result grid(boolean[] blocked, int width, int height, int start, int goal, Method method,
            Heuristic heuristic, double weight, boolean diagonal) {
        return grid(blocked, null, width, height, start, goal, method, heuristic, weight, diagonal);
    }

    /**
     * Searches a grid whose entered cells cost {@code 1 + extra[cell]} times the step length
     * (a guidance cost map, as Neural A* learns).
     *
     * @param blocked per cell whether it is an obstacle
     * @param extra per-cell extra cost, or {@code null}
     * @param width the width
     * @param height the height
     * @param start the start cell
     * @param goal the goal cell
     * @param method the algorithm (not JPS when {@code extra} is given)
     * @param heuristic the heuristic
     * @param weight {@code w} for weighted A*
     * @param diagonal 8-connected moves
     * @return the result
     */
    public static Result grid(boolean[] blocked, double[] extra, int width, int height, int start, int goal,
            Method method, Heuristic heuristic, double weight, boolean diagonal) {
        if (method == Method.JPS) {
            if (!diagonal || extra != null) {
                throw new IllegalArgumentException("JPS needs a uniform-cost 8-connected grid");
            }
            return jps(blocked, width, height, start, goal, heuristic == Heuristic.ZERO ? Heuristic.OCTILE : heuristic);
        }
        int n = width * height;
        double[] g = new double[n];
        Arrays.fill(g, Double.POSITIVE_INFINITY);
        int[] parent = new int[n];
        Arrays.fill(parent, -1);
        boolean[] closed = new boolean[n];
        int gx = goal % width;
        int gy = goal / width;
        double hw = switch (method) {
            case BFS, DIJKSTRA -> 0;
            case WEIGHTED_ASTAR -> weight;
            default -> 1;
        };
        boolean greedy = method == Method.GREEDY;
        PriorityQueue<Node> open = queue();
        g[start] = 0;
        open.add(new Node(start, hw * heuristic(heuristic, start % width, start / width, gx, gy), 0));
        int expanded = 0;
        int generated = 1;
        int[][] moves = diagonal
            ? new int[][] {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
            : new int[][] {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        long order = 0;
        double[] bfsOrder = method == Method.BFS ? new double[n] : null;
        while (!open.isEmpty()) {
            Node cur = open.poll();
            int c = cur.cell;
            if (closed[c] || (!greedy && cur.g > g[c])) {
                continue;
            }
            closed[c] = true;
            expanded++;
            if (c == goal) {
                break;
            }
            int x = c % width;
            int y = c / width;
            for (int[] m : moves) {
                int nx = x + m[0];
                int ny = y + m[1];
                if (!free(blocked, width, height, nx, ny)) {
                    continue;
                }
                if (m[0] != 0 && m[1] != 0 && (!free(blocked, width, height, x + m[0], y)
                        || !free(blocked, width, height, x, y + m[1]))) {
                    continue;
                }
                int nc = ny * width + nx;
                if (closed[nc]) {
                    continue;
                }
                double step = m[0] != 0 && m[1] != 0 ? Math.sqrt(2) : 1;
                if (extra != null) {
                    step *= 1 + extra[nc];
                }
                double ng = g[c] + step;
                if (ng < g[nc]) {
                    g[nc] = ng;
                    parent[nc] = c;
                    double h = heuristic(heuristic, nx, ny, gx, gy);
                    double f;
                    if (method == Method.BFS) {
                        bfsOrder[nc] = ++order;
                        f = bfsOrder[nc];
                    } else {
                        f = greedy ? h : ng + hw * h;
                    }
                    open.add(new Node(nc, f, method == Method.BFS ? 0 : ng));
                    generated++;
                }
            }
        }
        return finish(parent, g, start, goal, expanded, generated);
    }

    private static boolean free(boolean[] blocked, int width, int height, int x, int y) {
        return x >= 0 && y >= 0 && x < width && y < height && !blocked[y * width + x];
    }

    private static Result finish(int[] parent, double[] g, int start, int goal, int expanded, int generated) {
        if (Double.isInfinite(g[goal])) {
            return new Result(List.of(), Double.POSITIVE_INFINITY, expanded, generated);
        }
        List<Integer> path = new ArrayList<>();
        for (int c = goal; c != -1; c = c == start ? -1 : parent[c]) {
            path.add(c);
        }
        Collections.reverse(path);
        return new Result(path, g[goal], expanded, generated);
    }

    // ---- jump point search ---------------------------------------------------------

    private static Result jps(boolean[] blocked, int width, int height, int start, int goal, Heuristic heuristic) {
        int n = width * height;
        double[] g = new double[n];
        Arrays.fill(g, Double.POSITIVE_INFINITY);
        int[] parent = new int[n];
        Arrays.fill(parent, -1);
        boolean[] closed = new boolean[n];
        int gx = goal % width;
        int gy = goal / width;
        PriorityQueue<Node> open = queue();
        g[start] = 0;
        open.add(new Node(start, heuristic(heuristic, start % width, start / width, gx, gy), 0));
        int expanded = 0;
        int generated = 1;
        Jumper j = new Jumper(blocked, width, height, goal);
        while (!open.isEmpty()) {
            Node cur = open.poll();
            int c = cur.cell;
            if (closed[c] || cur.g > g[c]) {
                continue;
            }
            closed[c] = true;
            expanded++;
            if (c == goal) {
                break;
            }
            int x = c % width;
            int y = c / width;
            for (int[] nb : j.neighbours(x, y, parent[c])) {
                int jp = j.jump(nb[0], nb[1], x, y);
                if (jp < 0 || closed[jp]) {
                    continue;
                }
                int jx = jp % width;
                int jy = jp / width;
                double ng = g[c] + heuristic(Heuristic.OCTILE, x, y, jx, jy);
                if (ng < g[jp] - 1e-12) {
                    g[jp] = ng;
                    parent[jp] = c;
                    open.add(new Node(jp, ng + heuristic(heuristic, jx, jy, gx, gy), ng));
                    generated++;
                }
            }
        }
        Result r = finish(parent, g, start, goal, expanded, generated);
        if (!r.found()) {
            return r;
        }
        List<Integer> full = new ArrayList<>();
        List<Integer> jumps = r.path();
        full.add(jumps.get(0));
        for (int k = 1; k < jumps.size(); k++) {
            int a = jumps.get(k - 1);
            int b = jumps.get(k);
            int ax = a % width;
            int ay = a / width;
            int dx = Integer.signum(b % width - ax);
            int dy = Integer.signum(b / width - ay);
            while (ax != b % width || ay != b / width) {
                ax += dx;
                ay += dy;
                full.add(ay * width + ax);
            }
        }
        return new Result(full, r.cost(), r.expanded(), r.generated());
    }

    /** Jump point search's jumps and pruned neighbours, for moves that never cut corners. */
    private record Jumper(boolean[] blocked, int width, int height, int goal) {

        boolean ok(int x, int y) {
            return free(blocked, width, height, x, y);
        }

        List<int[]> neighbours(int x, int y, int parentCell) {
            List<int[]> out = new ArrayList<>(8);
            if (parentCell < 0) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        if (ok(x + dx, y + dy) && (dx == 0 || dy == 0 || (ok(x + dx, y) && ok(x, y + dy)))) {
                            out.add(new int[] {x + dx, y + dy});
                        }
                    }
                }
                return out;
            }
            int dx = Integer.signum(x - parentCell % width);
            int dy = Integer.signum(y - parentCell / width);
            if (dx != 0 && dy != 0) {
                boolean v = ok(x, y + dy);
                boolean h = ok(x + dx, y);
                if (v) {
                    out.add(new int[] {x, y + dy});
                }
                if (h) {
                    out.add(new int[] {x + dx, y});
                }
                if (v && h) {
                    out.add(new int[] {x + dx, y + dy});
                }
            } else if (dx != 0) {
                boolean next = ok(x + dx, y);
                boolean top = ok(x, y + 1);
                boolean bottom = ok(x, y - 1);
                if (next) {
                    out.add(new int[] {x + dx, y});
                    if (top) {
                        out.add(new int[] {x + dx, y + 1});
                    }
                    if (bottom) {
                        out.add(new int[] {x + dx, y - 1});
                    }
                }
                if (top) {
                    out.add(new int[] {x, y + 1});
                }
                if (bottom) {
                    out.add(new int[] {x, y - 1});
                }
            } else {
                boolean next = ok(x, y + dy);
                boolean right = ok(x + 1, y);
                boolean left = ok(x - 1, y);
                if (next) {
                    out.add(new int[] {x, y + dy});
                    if (right) {
                        out.add(new int[] {x + 1, y + dy});
                    }
                    if (left) {
                        out.add(new int[] {x - 1, y + dy});
                    }
                }
                if (right) {
                    out.add(new int[] {x + 1, y});
                }
                if (left) {
                    out.add(new int[] {x - 1, y});
                }
            }
            return out;
        }

        int jump(int x, int y, int px, int py) {
            int dx = x - px;
            int dy = y - py;
            while (true) {
                if (!ok(x, y)) {
                    return -1;
                }
                if (dx != 0 && dy != 0 && (!ok(x - dx, y) || !ok(x, y - dy))) {
                    return -1;
                }
                if (y * width + x == goal) {
                    return goal;
                }
                if (dx != 0 && dy != 0) {
                    if (jump(x + dx, y, x, y) >= 0 || jump(x, y + dy, x, y) >= 0) {
                        return y * width + x;
                    }
                } else if (dx != 0) {
                    if ((ok(x, y - 1) && !ok(x - dx, y - 1)) || (ok(x, y + 1) && !ok(x - dx, y + 1))) {
                        return y * width + x;
                    }
                } else {
                    if ((ok(x - 1, y) && !ok(x - 1, y - dy)) || (ok(x + 1, y) && !ok(x + 1, y - dy))) {
                        return y * width + x;
                    }
                }
                x += dx;
                y += dy;
            }
        }
    }

    // ---- generic implicit graphs ---------------------------------------------------

    /**
     * Best-first search on an implicit graph with unit edge costs.
     *
     * @param <S> the state type
     * @param start the start state
     * @param goal whether a state is a goal
     * @param successors the neighbours of a state
     * @param key a hashable key of a state (states themselves may be arrays)
     * @param heuristic the heuristic
     * @param method BFS, DIJKSTRA, ASTAR, WEIGHTED_ASTAR or GREEDY
     * @param weight {@code w} for weighted A*
     * @param maxExpansions a budget
     * @return the result; {@code path} holds indices into {@code states} of the result's
     *     {@link GenericResult}
     */
    public static <S> GenericResult<S> search(S start, Predicate<S> goal, Function<S, List<S>> successors,
            Function<S, Object> key, ToDoubleFunction<S> heuristic, Method method, double weight, int maxExpansions) {
        double hw = switch (method) {
            case BFS, DIJKSTRA -> 0;
            case WEIGHTED_ASTAR -> weight;
            case JPS -> throw new IllegalArgumentException("JPS is grid-only");
            default -> 1;
        };
        boolean greedy = method == Method.GREEDY;
        List<S> states = new ArrayList<>();
        Map<Object, Integer> index = new HashMap<>();
        List<Double> g = new ArrayList<>();
        List<Integer> parent = new ArrayList<>();
        List<Boolean> closed = new ArrayList<>();
        states.add(start);
        index.put(key.apply(start), 0);
        g.add(0.0);
        parent.add(-1);
        closed.add(false);
        PriorityQueue<Node> open = queue();
        open.add(new Node(0, hw == 0 && !greedy ? 0 : heuristic.applyAsDouble(start) * (greedy ? 1 : hw), 0));
        int expanded = 0;
        int generated = 1;
        int found = -1;
        while (!open.isEmpty() && expanded < maxExpansions) {
            Node cur = open.poll();
            int c = cur.cell;
            if (closed.get(c) || (!greedy && cur.g > g.get(c))) {
                continue;
            }
            closed.set(c, true);
            expanded++;
            S s = states.get(c);
            if (goal.test(s)) {
                found = c;
                break;
            }
            for (S t : successors.apply(s)) {
                Object k = key.apply(t);
                Integer id = index.get(k);
                double ng = g.get(c) + 1;
                if (id == null) {
                    id = states.size();
                    states.add(t);
                    index.put(k, id);
                    g.add(Double.POSITIVE_INFINITY);
                    parent.add(-1);
                    closed.add(false);
                }
                if (closed.get(id) || ng >= g.get(id)) {
                    continue;
                }
                g.set(id, ng);
                parent.set(id, c);
                double h = hw == 0 && !greedy ? 0 : heuristic.applyAsDouble(t);
                open.add(new Node(id, greedy ? h : ng + hw * h, ng));
                generated++;
            }
        }
        if (found < 0) {
            return new GenericResult<>(List.of(), Double.POSITIVE_INFINITY, expanded, generated);
        }
        List<S> path = new ArrayList<>();
        for (int c = found; c != -1; c = parent.get(c)) {
            path.add(states.get(c));
        }
        Collections.reverse(path);
        return new GenericResult<>(path, g.get(found), expanded, generated);
    }

    /**
     * A result of {@link #search}.
     *
     * @param <S> the state type
     * @param path the states from start to goal, empty if none
     * @param cost the path cost
     * @param expanded nodes expanded
     * @param generated nodes pushed
     */
    public record GenericResult<S>(List<S> path, double cost, int expanded, int generated) {

        /** @return whether a path was found */
        public boolean found() {
            return !path.isEmpty();
        }
    }
}
