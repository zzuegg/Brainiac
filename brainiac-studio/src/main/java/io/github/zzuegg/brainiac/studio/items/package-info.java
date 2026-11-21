/**
 * The things a user makes in the studio (conversations, agents, graphs, projects), seen through
 * one interface so every client can list, rename, move, duplicate and delete them the same way.
 * Deleting moves an item to the {@linkplain io.github.zzuegg.brainiac.studio.items.Trash trash},
 * from where it can be restored for 30 days; {@linkplain io.github.zzuegg.brainiac.studio.items.Owners
 * owners} record which project an item belongs to when its own file does not say.
 */
package io.github.zzuegg.brainiac.studio.items;
