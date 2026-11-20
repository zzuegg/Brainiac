package io.github.zzuegg.brainiac.studio.agent;

import java.util.Map;

/**
 * Names tools had before they were renamed. Roles, rules and project settings saved with an old
 * name keep working, and a model that calls a tool by its old name reaches the new one.
 */
public final class ToolNames {

    /** Old names and the names that replaced them. */
    public static final Map<String, String> RENAMED = Map.of(
            "read_project_file", "read_file",
            "list_project_files", "list_files",
            "search_files", "grep");

    private ToolNames() {
    }

    /**
     * The current name of a tool.
     *
     * @param name a name, perhaps an old one
     * @return the name it has now
     */
    public static String current(String name) {
        return name == null ? null : RENAMED.getOrDefault(name, name);
    }
}
