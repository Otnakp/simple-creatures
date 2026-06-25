#include "world.h"
#include <stdlib.h>

void world_init(World *w, int width, int height)
{
    w->width = width;
    w->height = height;
    w->grid = malloc(sizeof(int) * width * height);
    for (int i = 0; i < width * height; i++)
        w->grid[i] = CELL_EMPTY;
}

void world_populate(World *w, int food_count, int poison_count)
{
    for (int i = 0; i < food_count; i++)
        w->grid[rand() % (w->width * w->height)] = CELL_FOOD;
    for (int i = 0; i < poison_count; i++)
        w->grid[rand() % (w->width * w->height)] = CELL_POISON;
}

CellType world_get(World *w, int x, int y)
{
    if (x < 0 || y < 0 || x >= w->width || y >= w->height)
        return CELL_EMPTY;
    return w->grid[y * w->width + x];
}

void world_set(World *w, int x, int y, CellType type)
{
    if (x < 0 || y < 0 || x >= w->width || y >= w->height)
        return;
    w->grid[y * w->width + x] = type;
}

void world_free(World *w)
{
    free(w->grid);
}