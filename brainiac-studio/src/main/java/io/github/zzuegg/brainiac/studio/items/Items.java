package io.github.zzuegg.brainiac.studio.items;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * Every kind of item behind one set of actions: list by project, rename, move to another
 * project, duplicate, delete to the trash and restore. Both of the studio's pages use it, so an
 * item behaves the same wherever it is shown.
 *
 * <p>An item belongs to the project its store records ({@link ItemKind#keepsProject}) or else
 * the one {@link Owners} records. An item whose project no longer exists (it is in the trash)
 * shows in the default project until that project comes back.
 */
public final class Items {

    /** The project every item without another belongs to. */
    public static final String DEFAULT = "default";

    private static final System.Logger LOG = System.getLogger(Items.class.getName());

    private final Map<String, ItemKind> kinds = new LinkedHashMap<>();
    private final Owners owners;
    private final Trash trash;
    private final Predicate<String> projectExists;

    /**
     * Makes the registry; kinds are {@linkplain #register registered} after.
     *
     * @param owners the projects of items whose store keeps none
     * @param trash where deleted items go
     * @param projectExists whether a project id names a project now
     */
    public Items(Owners owners, Trash trash, Predicate<String> projectExists) {
        this.owners = owners;
        this.trash = trash;
        this.projectExists = projectExists;
    }

    /**
     * Adds a kind.
     *
     * @param kind the kind
     * @return this
     */
    public synchronized Items register(ItemKind kind) {
        kinds.put(kind.kind(), kind);
        return this;
    }

    /** @return the owners */
    public Owners owners() {
        return owners;
    }

    /**
     * Returns a kind.
     *
     * @param name its name
     * @return the kind
     * @throws IllegalArgumentException if there is no such kind
     */
    public synchronized ItemKind kind(String name) {
        ItemKind k = kinds.get(name);
        if (k == null) {
            throw new IllegalArgumentException("no kind of item " + name + ": use " + String.join(", ", kinds.keySet()));
        }
        return k;
    }

    /**
     * Lists items, the most recently changed first.
     *
     * @param project only the items of this project, or null for every project
     * @param only only these kinds, or null or empty for every kind but projects
     * @return the items, each with the project it belongs to
     * @throws IOException if a store cannot be read
     */
    public List<Item> list(String project, Collection<String> only) throws IOException {
        List<ItemKind> ks;
        synchronized (this) {
            ks = only == null || only.isEmpty()
                    ? kinds.values().stream().filter(k -> !ItemKind.PROJECT.equals(k.kind())).toList()
                    : only.stream().map(this::kind).toList();
        }
        List<Item> out = new ArrayList<>();
        for (ItemKind k : ks) {
            for (Item i : k.list()) {
                Item placed = place(k, i);
                if (project == null || placed.project() == null || project.equals(placed.project())) {
                    out.add(placed);
                }
            }
        }
        out.sort(Comparator.comparingLong(Item::updated).reversed());
        return out;
    }

    /**
     * Finds one item.
     *
     * @param kind its kind
     * @param id its id
     * @return the item
     * @throws IllegalArgumentException if there is no such item
     * @throws IOException if it cannot be read
     */
    public Item get(String kind, String id) throws IOException {
        ItemKind k = kind(kind);
        return place(k, k.get(id).orElseThrow(() -> new IllegalArgumentException("there is no " + kind + " " + id)));
    }

    /**
     * Renames an item.
     *
     * @param kind its kind
     * @param id its id
     * @param title the new name
     * @return the item afterwards
     * @throws IOException if the change cannot be kept
     */
    public Item rename(String kind, String id, String title) throws IOException {
        if (title == null || title.isBlank()) {
            throw new IllegalArgumentException("a name cannot be empty");
        }
        ItemKind k = kind(kind);
        get(kind, id);
        String now = k.rename(id, title.strip());
        if (!now.equals(id)) {
            owners.rekey(kind, id, now);
        }
        return get(kind, now);
    }

    /**
     * Moves an item to another project.
     *
     * @param kind its kind
     * @param id its id
     * @param project the project's id
     * @return the item afterwards
     * @throws IOException if the change cannot be kept
     */
    public Item move(String kind, String id, String project) throws IOException {
        if (project == null || !projectExists.test(project)) {
            throw new IllegalArgumentException("there is no project " + project);
        }
        ItemKind k = kind(kind);
        Item i = get(kind, id);
        if (i.project() == null) {
            throw new IllegalArgumentException("a " + kind + " does not belong to a project");
        }
        if (k.keepsProject()) {
            k.move(id, project);
        } else {
            owners.set(kind, id, project);
        }
        return get(kind, id);
    }

    /**
     * Copies an item into the same project.
     *
     * @param kind its kind
     * @param id its id
     * @return the copy
     * @throws IOException if it cannot be written
     */
    public Item duplicate(String kind, String id) throws IOException {
        ItemKind k = kind(kind);
        Item i = get(kind, id);
        String copy = k.duplicate(id);
        if (!k.keepsProject() && i.project() != null) {
            owners.set(kind, copy, i.project());
        }
        return get(kind, copy);
    }

    /**
     * Moves an item to the trash.
     *
     * @param kind its kind
     * @param id its id
     * @return the trash entry, whose id {@link #restore} takes
     * @throws IOException if it cannot be moved
     */
    public Trash.Entry delete(String kind, String id) throws IOException {
        ItemKind k = kind(kind);
        return trash.put(get(kind, id), k);
    }

    /**
     * Puts a deleted item back, in the project it was in if that still exists.
     *
     * @param entry the trash entry's id
     * @return the item
     * @throws IOException if it cannot be put back
     */
    public Item restore(String entry) throws IOException {
        Trash.Entry e = trash.get(entry).orElseThrow(() -> new IllegalArgumentException("nothing in the trash is called " + entry));
        ItemKind k = kind(e.item().kind());
        trash.restore(entry, k);
        return get(k.kind(), e.item().id());
    }

    /**
     * Lists the trash, after deleting what was kept long enough.
     *
     * @return the entries, the most recently deleted first
     */
    public List<Trash.Entry> trash() {
        for (Trash.Entry e : trash.expired()) {
            try {
                purge(e.id());
            } catch (IOException | RuntimeException x) {
                LOG.log(System.Logger.Level.WARNING, "cannot empty " + e.id() + " from the trash", x);
            }
        }
        return trash.list();
    }

    /**
     * Deletes a trash entry for good.
     *
     * @param entry the entry's id
     * @return whether there was one
     * @throws IOException if it cannot be deleted
     */
    public boolean purge(String entry) throws IOException {
        Trash.Entry e = trash.get(entry).orElse(null);
        if (e == null) {
            return false;
        }
        ItemKind k;
        synchronized (this) {
            k = kinds.get(e.item().kind());
        }
        boolean had = trash.purge(entry, k);
        owners.forget(e.item().kind(), e.item().id());
        return had;
    }

    private Item place(ItemKind k, Item i) {
        if (ItemKind.PROJECT.equals(k.kind())) {
            return i.in(null);
        }
        String p = k.keepsProject() ? i.project() : owners.of(k.kind(), i.id());
        return i.in(p == null || !projectExists.test(p) ? DEFAULT : p);
    }
}
