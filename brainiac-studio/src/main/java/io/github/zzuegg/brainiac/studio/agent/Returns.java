package io.github.zzuegg.brainiac.studio.agent;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * What a tool returns, in a few words, such as {@code {file, characters, mode}}. Models read it in
 * the tool's description, and scripts from {@code run_script} rely on it to use the result. Every
 * tool says it.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Returns {

    /**
     * @return the shape of the result: an object's keys in braces, a list in brackets, or text
     */
    String value();
}
