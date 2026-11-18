package io.github.zzuegg.brainiac.experimental.rl;

/**
 * Kuhn poker (Kuhn, 1950), the smallest interesting poker game and the standard first test
 * of imperfect-information game solvers. The deck is Jack, Queen and King; each player antes
 * one chip and gets one card. Player 0 checks or bets one chip; player 1 then checks or bets
 * (after a check) or folds or calls (after a bet); after check and bet, player 0 folds or
 * calls. A showdown gives the pot to the higher card.
 *
 * <p>Action 0 is check or fold, action 1 bet or call. The game has 12 information sets, and a
 * one-parameter family of Nash equilibria in which player 0 wins {@code -1/18} per hand on
 * average. Features are the card (one-hot, 3) and the two betting slots (one-hot, 2 x 2).
 */
public final class KuhnPoker implements ExtensiveFormGame {

    /** The game value for player 0 at any Nash equilibrium. */
    public static final double VALUE = -1.0 / 18;

    private static final int[][] DEALS = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}};
    private static final char[] RANKS = {'J', 'Q', 'K'};

    /** Creates the game. */
    public KuhnPoker() {
    }

    @Override
    public State root() {
        return new Node(-1, -1, "");
    }

    @Override
    public int maxActions() {
        return 2;
    }

    @Override
    public int featureSize() {
        return 7;
    }

    @Override
    public String toString() {
        return "Kuhn poker";
    }

    private record Node(int card0, int card1, String history) implements State {

        @Override
        public boolean terminal() {
            return history.equals("pp") || history.equals("bp") || history.equals("bb") || history.equals("pbp")
                    || history.equals("pbb");
        }

        @Override
        public int player() {
            return card0 < 0 ? CHANCE : history.length() % 2;
        }

        @Override
        public int actionCount() {
            return card0 < 0 ? DEALS.length : 2;
        }

        @Override
        public boolean legal(int action) {
            return card0 < 0 ? action >= 0 && action < DEALS.length : action == 0 || action == 1;
        }

        @Override
        public double chanceProbability(int action) {
            return 1.0 / DEALS.length;
        }

        @Override
        public State play(int action) {
            if (card0 < 0) {
                return new Node(DEALS[action][0], DEALS[action][1], "");
            }
            return new Node(card0, card1, history + (action == 0 ? 'p' : 'b'));
        }

        @Override
        public double payoff() {
            int sign = card0 > card1 ? 1 : -1;
            return switch (history) {
                case "pp" -> sign;
                case "bp" -> 1;
                case "pbp" -> -1;
                default -> 2 * sign;
            };
        }

        @Override
        public String infoSet() {
            return RANKS[player() == 0 ? card0 : card1] + ":" + history;
        }

        @Override
        public float[] features() {
            float[] f = new float[7];
            f[player() == 0 ? card0 : card1] = 1;
            for (int i = 0; i < history.length() && i < 2; i++) {
                f[3 + 2 * i + (history.charAt(i) == 'p' ? 0 : 1)] = 1;
            }
            return f;
        }
    }
}
