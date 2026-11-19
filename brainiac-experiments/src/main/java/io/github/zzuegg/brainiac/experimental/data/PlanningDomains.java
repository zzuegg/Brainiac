package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.symbolic.StripsTask;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Size-parametrised generators of grounded STRIPS tasks ({@link StripsTask}) for the classical
 * IPC-style toy domains: Blocksworld (four operators), Gripper, a Logistics-lite (one truck per
 * city, one plane, two locations per city of which one is the airport), VisitAll on a grid,
 * Ferry, and the 8-puzzle. Every generator declares the natural mutex groups of its domain (the
 * invariants a translator such as Fast Downward's would find), so the tasks also have a SAS+
 * view. Small sizes have exactly computable optimal costs; the learned planners train on those
 * and are tested on instances two to four times larger.
 */
public final class PlanningDomains {

    private PlanningDomains() {
    }

    // ---- Blocksworld ---------------------------------------------------------------------

    /**
     * A random Blocksworld task: the initial state and the goal are random tower configurations
     * (every block is put on the table or on a random existing tower).
     *
     * @param blocks the number of blocks
     * @param seed the seed
     * @return the task
     */
    public static StripsTask blocksworld(int blocks, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] below0 = randomTowers(blocks, r);
        int[] below1;
        do {
            below1 = randomTowers(blocks, r);
        } while (java.util.Arrays.equals(below0, below1) && blocks > 1);
        return blocksworld(below0, below1, "blocksworld-" + blocks + "-" + seed);
    }

    /**
     * Random towers: {@code below[x]} is the block under {@code x}, or -1 for the table.
     */
    private static int[] randomTowers(int n, SplittableRandom r) {
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        int[] below = new int[n];
        List<Integer> tops = new ArrayList<>();
        for (int x : order) {
            int k = r.nextInt(tops.size() + 1);
            if (k == tops.size()) {
                below[x] = -1;
                tops.add(x);
            } else {
                below[x] = tops.get(k);
                tops.set(k, x);
            }
        }
        return below;
    }

    /**
     * A Blocksworld task between two given configurations.
     *
     * @param initBelow per block the block below it in the initial state, -1 for the table
     * @param goalBelow the same for the goal (every block's position is a goal)
     * @param name the instance name
     * @return the task
     */
    public static StripsTask blocksworld(int[] initBelow, int[] goalBelow, String name) {
        int n = initBelow.length;
        StripsTask.Builder b = StripsTask.builder(name);
        String[] bl = new String[n];
        for (int i = 0; i < n; i++) {
            bl[i] = "b" + i;
            b.object(bl[i]);
        }
        int handempty = b.fact("handempty");
        int[] clear = new int[n];
        int[] table = new int[n];
        int[] holding = new int[n];
        int[][] on = new int[n][n];
        for (int x = 0; x < n; x++) {
            clear[x] = b.fact("clear", bl[x]);
            table[x] = b.fact("ontable", bl[x]);
            holding[x] = b.fact("holding", bl[x]);
        }
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                if (x != y) {
                    on[x][y] = b.fact("on", bl[x], bl[y]);
                }
            }
        }
        for (int x = 0; x < n; x++) {
            b.action("pickup", new String[] {bl[x]}, new int[] {clear[x], table[x], handempty},
                new int[] {holding[x]}, new int[] {clear[x], table[x], handempty}, 1);
            b.action("putdown", new String[] {bl[x]}, new int[] {holding[x]},
                new int[] {clear[x], table[x], handempty}, new int[] {holding[x]}, 1);
            for (int y = 0; y < n; y++) {
                if (x == y) {
                    continue;
                }
                b.action("stack", new String[] {bl[x], bl[y]}, new int[] {holding[x], clear[y]},
                    new int[] {on[x][y], clear[x], handempty}, new int[] {holding[x], clear[y]}, 1);
                b.action("unstack", new String[] {bl[x], bl[y]}, new int[] {on[x][y], clear[x], handempty},
                    new int[] {holding[x], clear[y]}, new int[] {on[x][y], clear[x], handempty}, 1);
            }
        }
        for (int x = 0; x < n; x++) {
            int[] g1 = new int[n + 1];
            int[] g2 = new int[n + 1];
            g1[0] = table[x];
            g1[1] = holding[x];
            g2[0] = clear[x];
            g2[1] = holding[x];
            int k = 2;
            for (int y = 0; y < n; y++) {
                if (y != x) {
                    g1[k] = on[x][y];
                    g2[k] = on[y][x];
                    k++;
                }
            }
            b.mutex(g1);
            b.mutex(g2);
        }
        int[] hand = new int[n + 1];
        hand[0] = handempty;
        System.arraycopy(holding, 0, hand, 1, n);
        b.mutex(hand);
        List<Integer> init = new ArrayList<>();
        init.add(handempty);
        boolean[] covered = new boolean[n];
        for (int x = 0; x < n; x++) {
            init.add(initBelow[x] < 0 ? table[x] : on[x][initBelow[x]]);
            if (initBelow[x] >= 0) {
                covered[initBelow[x]] = true;
            }
        }
        for (int x = 0; x < n; x++) {
            if (!covered[x]) {
                init.add(clear[x]);
            }
        }
        int[] goal = new int[n];
        for (int x = 0; x < n; x++) {
            goal[x] = goalBelow[x] < 0 ? table[x] : on[x][goalBelow[x]];
        }
        return b.build(init.stream().mapToInt(Integer::intValue).toArray(), goal);
    }

    // ---- Gripper ---------------------------------------------------------------------------

    /**
     * Gripper: a robot with two grippers moves every ball from room A to room B.
     *
     * @param balls the number of balls
     * @return the task
     */
    public static StripsTask gripper(int balls) {
        StripsTask.Builder b = StripsTask.builder("gripper-" + balls);
        String[] rooms = {"roomA", "roomB"};
        String[] grippers = {"left", "right"};
        int[] robby = {b.fact("at-robby", rooms[0]), b.fact("at-robby", rooms[1])};
        int[] free = {b.fact("free", grippers[0]), b.fact("free", grippers[1])};
        int[][] at = new int[balls][2];
        int[][] carry = new int[balls][2];
        for (int i = 0; i < balls; i++) {
            for (int k = 0; k < 2; k++) {
                at[i][k] = b.fact("at", "ball" + i, rooms[k]);
                carry[i][k] = b.fact("carry", "ball" + i, grippers[k]);
            }
        }
        for (int k = 0; k < 2; k++) {
            b.action("move", new String[] {rooms[k], rooms[1 - k]}, new int[] {robby[k]}, new int[] {robby[1 - k]},
                new int[] {robby[k]}, 1);
        }
        for (int i = 0; i < balls; i++) {
            for (int room = 0; room < 2; room++) {
                for (int g = 0; g < 2; g++) {
                    String[] args = {"ball" + i, rooms[room], grippers[g]};
                    b.action("pick", args, new int[] {at[i][room], robby[room], free[g]}, new int[] {carry[i][g]},
                        new int[] {at[i][room], free[g]}, 1);
                    b.action("drop", args, new int[] {carry[i][g], robby[room]}, new int[] {at[i][room], free[g]},
                        new int[] {carry[i][g]}, 1);
                }
            }
        }
        b.mutex(robby);
        for (int i = 0; i < balls; i++) {
            b.mutex(at[i][0], at[i][1], carry[i][0], carry[i][1]);
        }
        for (int g = 0; g < 2; g++) {
            int[] m = new int[balls + 1];
            m[0] = free[g];
            for (int i = 0; i < balls; i++) {
                m[i + 1] = carry[i][g];
            }
            b.mutex(m);
        }
        int[] init = new int[balls + 3];
        init[0] = robby[0];
        init[1] = free[0];
        init[2] = free[1];
        int[] goal = new int[balls];
        for (int i = 0; i < balls; i++) {
            init[i + 3] = at[i][0];
            goal[i] = at[i][1];
        }
        return b.build(init, goal);
    }

    // ---- Logistics-lite --------------------------------------------------------------------

    /**
     * Logistics-lite: every city has an airport and one other location and one truck; one plane
     * flies between airports; packages start and end at random locations.
     *
     * @param cities the number of cities (2-3 in the standard set)
     * @param trucksPerCity trucks per city
     * @param packages the number of packages
     * @param nonUnitCosts whether flying costs 4, driving 2 and loading 1 (otherwise all 1)
     * @param seed the seed
     * @return the task
     */
    public static StripsTask logistics(int cities, int trucksPerCity, int packages, boolean nonUnitCosts,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        StripsTask.Builder b = StripsTask.builder("logistics-" + cities + "-" + packages + "-" + seed);
        int locs = 2 * cities;
        String[] loc = new String[locs];
        for (int c = 0; c < cities; c++) {
            loc[2 * c] = "apt" + c;
            loc[2 * c + 1] = "loc" + c;
        }
        int trucks = cities * trucksPerCity;
        String[] veh = new String[trucks + 1];
        for (int t = 0; t < trucks; t++) {
            veh[t] = "truck" + t;
        }
        veh[trucks] = "plane";
        int[][] vat = new int[trucks + 1][locs];
        for (int t = 0; t < trucks; t++) {
            int c = t / trucksPerCity;
            vat[t][2 * c] = b.fact("at", veh[t], loc[2 * c]);
            vat[t][2 * c + 1] = b.fact("at", veh[t], loc[2 * c + 1]);
        }
        for (int c = 0; c < cities; c++) {
            vat[trucks][2 * c] = b.fact("at", "plane", loc[2 * c]);
        }
        int[][] pat = new int[packages][locs];
        int[][] pin = new int[packages][trucks + 1];
        for (int p = 0; p < packages; p++) {
            for (int l = 0; l < locs; l++) {
                pat[p][l] = b.fact("at", "pkg" + p, loc[l]);
            }
            for (int v = 0; v <= trucks; v++) {
                pin[p][v] = b.fact("in", "pkg" + p, veh[v]);
            }
        }
        double load = 1;
        double drive = nonUnitCosts ? 2 : 1;
        double fly = nonUnitCosts ? 4 : 1;
        for (int t = 0; t < trucks; t++) {
            int c = t / trucksPerCity;
            for (int k = 0; k < 2; k++) {
                int from = 2 * c + k;
                int to = 2 * c + 1 - k;
                b.action("drive", new String[] {veh[t], loc[from], loc[to]}, new int[] {vat[t][from]},
                    new int[] {vat[t][to]}, new int[] {vat[t][from]}, drive);
            }
        }
        for (int c1 = 0; c1 < cities; c1++) {
            for (int c2 = 0; c2 < cities; c2++) {
                if (c1 != c2) {
                    b.action("fly", new String[] {"plane", loc[2 * c1], loc[2 * c2]}, new int[] {vat[trucks][2 * c1]},
                        new int[] {vat[trucks][2 * c2]}, new int[] {vat[trucks][2 * c1]}, fly);
                }
            }
        }
        for (int p = 0; p < packages; p++) {
            for (int v = 0; v <= trucks; v++) {
                for (int l = 0; l < locs; l++) {
                    if (v < trucks && l / 2 != v / trucksPerCity) {
                        continue;
                    }
                    if (v == trucks && l % 2 != 0) {
                        continue;
                    }
                    String[] args = {"pkg" + p, veh[v], loc[l]};
                    b.action("load", args, new int[] {pat[p][l], vat[v][l]}, new int[] {pin[p][v]},
                        new int[] {pat[p][l]}, load);
                    b.action("unload", args, new int[] {pin[p][v], vat[v][l]}, new int[] {pat[p][l]},
                        new int[] {pin[p][v]}, load);
                }
            }
        }
        for (int p = 0; p < packages; p++) {
            int[] m = new int[locs + trucks + 1];
            System.arraycopy(pat[p], 0, m, 0, locs);
            System.arraycopy(pin[p], 0, m, locs, trucks + 1);
            b.mutex(m);
        }
        for (int t = 0; t < trucks; t++) {
            int c = t / trucksPerCity;
            b.mutex(vat[t][2 * c], vat[t][2 * c + 1]);
        }
        int[] plane = new int[cities];
        for (int c = 0; c < cities; c++) {
            plane[c] = vat[trucks][2 * c];
        }
        b.mutex(plane);
        List<Integer> init = new ArrayList<>();
        for (int t = 0; t < trucks; t++) {
            int c = t / trucksPerCity;
            init.add(vat[t][2 * c + r.nextInt(2)]);
        }
        init.add(vat[trucks][2 * r.nextInt(cities)]);
        int[] goal = new int[packages];
        for (int p = 0; p < packages; p++) {
            int from = r.nextInt(locs);
            int to;
            do {
                to = r.nextInt(locs);
            } while (to == from);
            init.add(pat[p][from]);
            goal[p] = pat[p][to];
        }
        return b.build(init.stream().mapToInt(Integer::intValue).toArray(), goal);
    }

    // ---- VisitAll ---------------------------------------------------------------------------

    /**
     * VisitAll: a robot on a {@code side x side} 4-connected grid must visit every cell.
     *
     * @param side the grid side
     * @return the task
     */
    public static StripsTask visitAll(int side) {
        StripsTask.Builder b = StripsTask.builder("visitall-" + side);
        int n = side * side;
        int[] at = new int[n];
        int[] visited = new int[n];
        for (int c = 0; c < n; c++) {
            at[c] = b.fact("at-robot", "c" + c);
            visited[c] = b.fact("visited", "c" + c);
        }
        for (int c = 0; c < n; c++) {
            int x = c % side;
            int y = c / side;
            int[][] d = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int[] m : d) {
                int nx = x + m[0];
                int ny = y + m[1];
                if (nx < 0 || ny < 0 || nx >= side || ny >= side) {
                    continue;
                }
                int t = ny * side + nx;
                b.action("move", new String[] {"c" + c, "c" + t}, new int[] {at[c]}, new int[] {at[t], visited[t]},
                    new int[] {at[c]}, 1);
            }
        }
        b.mutex(at);
        return b.build(new int[] {at[0], visited[0]}, visited);
    }

    // ---- Ferry ------------------------------------------------------------------------------

    /**
     * Ferry: a ferry with room for one car moves cars between locations.
     *
     * @param locations the number of locations
     * @param cars the number of cars
     * @param seed the seed
     * @return the task
     */
    public static StripsTask ferry(int locations, int cars, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        StripsTask.Builder b = StripsTask.builder("ferry-" + locations + "-" + cars + "-" + seed);
        int[] ferryAt = new int[locations];
        for (int l = 0; l < locations; l++) {
            ferryAt[l] = b.fact("at-ferry", "l" + l);
        }
        int empty = b.fact("empty-ferry");
        int[][] at = new int[cars][locations];
        int[] on = new int[cars];
        for (int c = 0; c < cars; c++) {
            for (int l = 0; l < locations; l++) {
                at[c][l] = b.fact("at", "car" + c, "l" + l);
            }
            on[c] = b.fact("on", "car" + c);
        }
        for (int l1 = 0; l1 < locations; l1++) {
            for (int l2 = 0; l2 < locations; l2++) {
                if (l1 != l2) {
                    b.action("sail", new String[] {"l" + l1, "l" + l2}, new int[] {ferryAt[l1]},
                        new int[] {ferryAt[l2]}, new int[] {ferryAt[l1]}, 1);
                }
            }
        }
        for (int c = 0; c < cars; c++) {
            for (int l = 0; l < locations; l++) {
                String[] args = {"car" + c, "l" + l};
                b.action("board", args, new int[] {at[c][l], ferryAt[l], empty}, new int[] {on[c]},
                    new int[] {at[c][l], empty}, 1);
                b.action("debark", args, new int[] {on[c], ferryAt[l]}, new int[] {at[c][l], empty},
                    new int[] {on[c]}, 1);
            }
        }
        b.mutex(ferryAt);
        int[] cap = new int[cars + 1];
        cap[0] = empty;
        System.arraycopy(on, 0, cap, 1, cars);
        b.mutex(cap);
        for (int c = 0; c < cars; c++) {
            int[] m = new int[locations + 1];
            System.arraycopy(at[c], 0, m, 0, locations);
            m[locations] = on[c];
            b.mutex(m);
        }
        List<Integer> init = new ArrayList<>();
        init.add(ferryAt[r.nextInt(locations)]);
        init.add(empty);
        int[] goal = new int[cars];
        for (int c = 0; c < cars; c++) {
            int from = r.nextInt(locations);
            int to;
            do {
                to = r.nextInt(locations);
            } while (to == from && locations > 1);
            init.add(at[c][from]);
            goal[c] = at[c][to];
        }
        return b.build(init.stream().mapToInt(Integer::intValue).toArray(), goal);
    }

    // ---- 8-puzzle ---------------------------------------------------------------------------

    /**
     * The sliding-tile puzzle as STRIPS, with the board layout of {@link PlanningMaps}
     * ({@code board[cell]} is the tile, 0 the blank; the goal has tiles {@code 1..n²-1} in order
     * and the blank last).
     *
     * @param board the initial board
     * @param side the side (3 for the 8-puzzle)
     * @return the task
     */
    public static StripsTask slidingTile(int[] board, int side) {
        int cells = side * side;
        StripsTask.Builder b = StripsTask.builder("tiles-" + side);
        int[][] at = new int[cells][cells];
        int[] blank = new int[cells];
        for (int c = 0; c < cells; c++) {
            blank[c] = b.fact("blank", "p" + c);
        }
        for (int t = 1; t < cells; t++) {
            for (int c = 0; c < cells; c++) {
                at[t][c] = b.fact("at", "t" + t, "p" + c);
            }
        }
        for (int c = 0; c < cells; c++) {
            int x = c % side;
            int y = c / side;
            int[][] d = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (int[] m : d) {
                int nx = x + m[0];
                int ny = y + m[1];
                if (nx < 0 || ny < 0 || nx >= side || ny >= side) {
                    continue;
                }
                int to = ny * side + nx;
                for (int t = 1; t < cells; t++) {
                    b.action("move", new String[] {"t" + t, "p" + c, "p" + to}, new int[] {at[t][c], blank[to]},
                        new int[] {at[t][to], blank[c]}, new int[] {at[t][c], blank[to]}, 1);
                }
            }
        }
        b.mutex(blank);
        for (int t = 1; t < cells; t++) {
            b.mutex(at[t]);
        }
        for (int c = 0; c < cells; c++) {
            int[] m = new int[cells];
            m[0] = blank[c];
            for (int t = 1; t < cells; t++) {
                m[t] = at[t][c];
            }
            b.mutex(m);
        }
        int[] init = new int[cells];
        for (int c = 0; c < cells; c++) {
            init[c] = board[c] == 0 ? blank[c] : at[board[c]][c];
        }
        int[] goal = new int[cells - 1];
        for (int t = 1; t < cells; t++) {
            goal[t - 1] = at[t][t - 1];
        }
        return b.build(init, goal);
    }
}
