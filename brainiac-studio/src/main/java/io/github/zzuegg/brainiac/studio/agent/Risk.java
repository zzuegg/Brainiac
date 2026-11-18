package io.github.zzuegg.brainiac.studio.agent;

import java.util.Locale;

/**
 * How much a tool call can change, cost or reach, from harmless to leaving the PC under the
 * user's name. Each role and mode decides per tier whether a call runs, waits for the user, or
 * is refused; {@link #OUTWARD} always waits, whatever the mode.
 */
public enum Risk {

    /** Looks at something: files, models, the GPU, the web. */
    READ("reads"),
    /** Keeps the GPU or CPU busy on this PC: drawing, speaking, training. */
    COMPUTE("uses this PC's GPU or CPU"),
    /** Costs money: a paid API, a rented machine. */
    SPEND("may cost money"),
    /** Makes something new in the project folder or the studio, overwriting nothing. */
    CREATE("creates a new file"),
    /** Changes or removes what is there: overwrites, edits, deletes. */
    WRITE("changes files"),
    /** Runs commands in the sandbox, or commits code. */
    CODE("runs commands in the sandbox"),
    /** Leaves the PC under the user's name: posts, mails. */
    OUTWARD("leaves this PC under your name");

    private final String what;

    Risk(String what) {
        this.what = what;
    }

    /**
     * Says what calls of this tier do, for approval cards.
     *
     * @return a few words
     */
    public String what() {
        return what;
    }

    /**
     * The tier's name in settings and JSON.
     *
     * @return {@code read}, {@code compute}, ...
     */
    public String key() {
        return name().toLowerCase(Locale.ROOT);
    }

    /**
     * Reads a tier's name.
     *
     * @param key {@code read}, {@code compute}, ...
     * @return the tier
     */
    public static Risk of(String key) {
        try {
            return valueOf(key.strip().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException | NullPointerException e) {
            throw new IllegalArgumentException("no risk tier " + key + ": use read, compute, spend, create, write, code "
                    + "or outward");
        }
    }

    /**
     * The riskier of two tiers.
     *
     * @param other another tier
     * @return the one further down the list
     */
    public Risk max(Risk other) {
        return other.ordinal() > ordinal() ? other : this;
    }

    /** What happens to a call of a tier. */
    public enum Policy {
        /** It runs. */
        ALLOW,
        /** It waits for the user. */
        ASK,
        /** It is refused. */
        DENY;

        /**
         * The policy's name in settings and JSON.
         *
         * @return {@code allow}, {@code ask} or {@code deny}
         */
        public String key() {
            return name().toLowerCase(Locale.ROOT);
        }

        /**
         * Reads a policy's name.
         *
         * @param key {@code allow}, {@code ask} or {@code deny}
         * @return the policy
         */
        public static Policy of(String key) {
            try {
                return valueOf(key.strip().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException | NullPointerException e) {
                throw new IllegalArgumentException("no policy " + key + ": use allow, ask or deny");
            }
        }

        /**
         * The stricter of two policies.
         *
         * @param other another policy
         * @return the one further down the list
         */
        public Policy strictest(Policy other) {
            return other.ordinal() > ordinal() ? other : this;
        }
    }

    /**
     * What a mode does with this tier before a role changes it: {@code ask} waits before anything
     * that uses the GPU, costs money, changes files, runs code or leaves the PC; {@code auto} runs
     * all of it except what leaves the PC. {@code plan} acts as {@code ask} once the plan is
     * approved; before that, everything but reading is refused.
     *
     * @param mode {@code plan}, {@code ask} or {@code auto}
     * @return the policy
     */
    public Policy byDefault(String mode) {
        if (this == READ || this == CREATE) {
            return Policy.ALLOW;
        }
        if (this == OUTWARD) {
            return Policy.ASK;
        }
        return "auto".equals(mode) ? Policy.ALLOW : Policy.ASK;
    }
}
