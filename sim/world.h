#ifndef WORLD_H
#define WORLD_H

typedef enum {
    CELL_EMPTY = 0,
    CELL_FOOD = 1,
    CELL_POISON = 2
} CellType;

typedef struct {
    int width;
    int height;
    int *grid;
} World;

void world_init(World *w, int width, int height);
void world_populate(World *w, int food_count, int poison_count);
CellType world_get(World *w, int x, int y);
void world_set(World *w, int x, int y, CellType type);
void world_free(World *w);

#endif