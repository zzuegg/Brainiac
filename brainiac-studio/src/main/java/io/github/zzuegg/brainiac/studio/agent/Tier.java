package io.github.zzuegg.brainiac.studio.agent;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Says how risky a tool is, and how its results are handled. A tool with neither this nor
 * {@link StudioTools.Acts} only reads.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Tier {

    /**
     * @return the tier of its calls; {@link RiskRefiner} can raise or lower it per call
     */
    Risk value();

    /**
     * @return what calling it does, for the approval card; empty for the tier's own words
     */
    String what() default "";

    /**
     * @return the kind of task it runs ({@code text}, {@code image}, {@code speech},
     *         {@code transcribe}, {@code video}): a call whose model is not on this PC costs
     *         money, so it counts as {@link Risk#SPEND}
     */
    String medium() default "";

    /**
     * @return true if it returns text from outside (web pages, mail, other agents): the agent is
     *         told to treat it as data, and while it holds such text, its calls that change things
     *         wait for the user even in auto mode
     */
    boolean untrusted() default false;

    /**
     * @return the most characters of its result the model sees, or 0 for the default
     */
    int resultChars() default 0;

    /**
     * @return true if it makes media (pictures, audio, clips) that count toward a run's budget
     */
    boolean media() default false;
}
