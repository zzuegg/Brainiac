package io.github.zzuegg.brainiac.studio.items;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * One kind of item and the store that keeps it. {@link Items} gives every kind the same actions;
 * a kind implements what it can and refuses the rest with an {@link IllegalArgumentException}
 * that says why.
 */
public interface ItemKind {

    /** Conversations. */
    String CHAT = "chat";
    /** Agent runs. */
    String AGENT = "agent";
    /** The builder's graphs. */
    String GRAPH = "graph";
    /** Projects themselves. */
    String PROJECT = "project";

    /** @return the kind's name, such as {@value #CHAT} */
    String kind();

    /**
     * Lists the items.
     *
     * @return them; an item's project is null unless the store itself keeps one
     * @throws IOException if they cannot be read
     */
    List<Item> list() throws IOException;

    /**
     * Finds one item.
     *
     * @param id its id
     * @return it, or empty
     * @throws IOException if it cannot be read
     */
    default Optional<Item> get(String id) throws IOException {
        return list().stream().filter(i -> i.id().equals(id)).findFirst();
    }

    /**
     * Renames an item.
     *
     * @param id its id
     * @param title the new name
     * @return its id afterwards, which changes for kinds named by their file
     * @throws IOException if the change cannot be kept
     */
    String rename(String id, String title) throws IOException;

    /**
     * Whether the store keeps each item's project itself; otherwise {@link Owners} does.
     *
     * @return true if {@link #move} is how items change project
     */
    default boolean keepsProject() {
        return false;
    }

    /**
     * Moves an item to another project, for kinds that {@linkplain #keepsProject keep the project}.
     *
     * @param id its id
     * @param project the project's id
     * @throws IOException if the change cannot be kept
     */
    default void move(String id, String project) throws IOException {
        throw new IllegalArgumentException(kind() + " items do not keep a project");
    }

    /**
     * Makes a copy of an item.
     *
     * @param id its id
     * @return the copy's id
     * @throws IOException if it cannot be written
     */
    default String duplicate(String id) throws IOException {
        throw new IllegalArgumentException("a " + kind() + " cannot be duplicated");
    }

    /**
     * Takes an item out of its store, moving its files into a folder.
     *
     * @param id its id
     * @param into an empty folder for its files
     * @return what {@link #attach} needs to put it back
     * @throws IOException if it cannot be moved
     */
    Map<String, Object> detach(String id, Path into) throws IOException;

    /**
     * Puts an item {@linkplain #detach taken out} back.
     *
     * @param id its id
     * @param from the folder its files were moved to
     * @param kept what {@link #detach} returned
     * @throws IOException if it cannot be put back
     */
    void attach(String id, Path from, Map<String, Object> kept) throws IOException;

    /**
     * Forgets anything kept elsewhere for an item that is gone for good. The trash deletes its files.
     *
     * @param id its id
     * @param kept what {@link #detach} returned
     * @throws IOException if it cannot be deleted
     */
    default void purge(String id, Map<String, Object> kept) throws IOException {
    }
}
