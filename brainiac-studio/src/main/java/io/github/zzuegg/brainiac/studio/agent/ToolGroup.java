package io.github.zzuegg.brainiac.studio.agent;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * The group a class of tools belongs to, such as {@code files} or {@code git}. Agents see the
 * groups with their summaries when tools load on demand, and {@code find_tools} loads a whole
 * group by its id.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface ToolGroup {

    /**
     * @return the group's id, short and lower case
     */
    String id();

    /**
     * @return what the group's tools are for, in a few words
     */
    String summary();
}
