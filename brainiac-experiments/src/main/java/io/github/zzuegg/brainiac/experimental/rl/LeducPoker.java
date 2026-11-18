package io.github.zzuegg.brainiac.experimental.rl;

/**
 * Leduc hold'em (Southey et al., 2005), the next step up from {@link KuhnPoker}: a deck of
 * two Jacks, two Queens and two Kings; each player antes one chip and gets one private card.
 * After a first betting round a public card is dealt, then a second round follows. Bets are
 * two chips in the first round and four in the second, with at most two raises per round, and
 * player 0 acts first in both. At showdown a player whose card pairs the public card wins;
 * otherwise the higher card wins, and equal ranks split the pot.
 *
 * <p>Actions are 0 fold (legal only when facing a raise), 1 check or call, and 2 raise (legal
 * while fewer than two raises were made in the round). Information sets are keyed by rank, which
 * merges suit-isomorphic ones losslessly: 288 information sets (936 with suits).
 * Features are the private rank and the public rank (one-hot, 3 each) and four betting slots
 * per round (one-hot call/raise, 2 each).
 *
 * <p>Southey et al. 2005, "Bayes' Bluff: Opponent Modelling in Poker",
 * <a href="https://arxiv.org/abs/1207.1411">arXiv:1207.1411</a>.
 */
public final class LeducPoker implements ExtensiveFormGame {

    private static final char[] RANKS = {'J', 'Q', 'K'};
    private static final int[] BET = {2, 4};

    /** Creates the game. */
    public LeducPoker() {
    }

    @Override
    public State root() {
        return new Node(-1, -1, -1, "", "");
    }

    @Override
    public int maxActions() {
        return 3;
    }

    @Override
    public int featureSize() {
        return 22;
    }

    @Override
    public String toString() {
        return "Leduc poker";
    }

    private static boolean roundOver(String h) {
        if (h.isEmpty()) {
            return false;
        }
        char last = h.charAt(h.length() - 1);
        return last == 'f' || last == 'c' && h.length() >= 2;
    }

    private static int raises(String h) {
        int n = 0;
        for (int i = 0; i < h.length(); i++) {
            n += h.charAt(i) == 'r' ? 1 : 0;
        }
        return n;
    }

    private record Node(int card0, int card1, int board, String round0, String round1) implements State {

        private int round() {
            return roundOver(round0) ? 1 : 0;
        }

        private String current() {
            return round() == 0 ? round0 : round1;
        }

        private boolean folded() {
            return round0.endsWith("f") || round1.endsWith("f");
        }

        @Override
        public boolean terminal() {
            return folded() || roundOver(round1);
        }

        @Override
        public int player() {
            if (card0 < 0 || roundOver(round0) && board < 0) {
                return CHANCE;
            }
            return current().length() % 2;
        }

        @Override
        public int actionCount() {
            if (card0 < 0) {
                return 30;
            }
            return board < 0 && roundOver(round0) ? 6 : 3;
        }

        @Override
        public boolean legal(int action) {
            if (card0 < 0) {
                return action >= 0 && action < 30;
            }
            if (player() == CHANCE) {
                return action >= 0 && action < 6 && action != card0 && action != card1;
            }
            String h = current();
            return switch (action) {
                case 0 -> h.endsWith("r");
                case 1 -> true;
                case 2 -> raises(h) < 2;
                default -> false;
            };
        }

        @Override
        public double chanceProbability(int action) {
            return card0 < 0 ? 1.0 / 30 : 1.0 / 4;
        }

        @Override
        public State play(int action) {
            if (card0 < 0) {
                int a = action / 5;
                int b = action % 5;
                return new Node(a, b >= a ? b + 1 : b, -1, "", "");
            }
            if (player() == CHANCE) {
                return new Node(card0, card1, action, round0, round1);
            }
            char c = action == 0 ? 'f' : action == 1 ? 'c' : 'r';
            return round() == 0 ? new Node(card0, card1, board, round0 + c, round1)
                    : new Node(card0, card1, board, round0, round1 + c);
        }

        /** Chips each player has put in, ante included. */
        private int[] contributions() {
            int[] in = {1, 1};
            String[] rounds = {round0, round1};
            for (int r = 0; r < 2; r++) {
                for (int i = 0; i < rounds[r].length(); i++) {
                    int p = i % 2;
                    char c = rounds[r].charAt(i);
                    if (c == 'c') {
                        in[p] = in[1 - p];
                    } else if (c == 'r') {
                        in[p] = in[1 - p] + BET[r];
                    }
                }
            }
            return in;
        }

        @Override
        public double payoff() {
            int[] in = contributions();
            String last = round1.isEmpty() ? round0 : round1;
            if (last.endsWith("f")) {
                int folder = (last.length() - 1) % 2;
                return folder == 0 ? -in[0] : in[1];
            }
            int r0 = card0 / 2;
            int r1 = card1 / 2;
            int rb = board / 2;
            int s0 = r0 == rb ? 10 + r0 : r0;
            int s1 = r1 == rb ? 10 + r1 : r1;
            return s0 > s1 ? in[1] : s0 < s1 ? -in[0] : 0;
        }

        @Override
        public String infoSet() {
            int own = player() == 0 ? card0 : card1;
            return "" + RANKS[own / 2] + (board < 0 ? '-' : RANKS[board / 2]) + ":" + round0 + "/" + round1;
        }

        @Override
        public float[] features() {
            float[] f = new float[22];
            int own = player() == 0 ? card0 : card1;
            f[own / 2] = 1;
            if (board >= 0) {
                f[3 + board / 2] = 1;
            }
            String[] rounds = {round0, round1};
            for (int r = 0; r < 2; r++) {
                for (int i = 0; i < rounds[r].length() && i < 4; i++) {
                    f[6 + 8 * r + 2 * i + (rounds[r].charAt(i) == 'r' ? 1 : 0)] = 1;
                }
            }
            return f;
        }
    }
}
