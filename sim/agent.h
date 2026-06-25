#ifndef AGENT_H
#define AGENT_H

#include "../network/network.h"
#include "world.h"

typedef enum {
    ACTION_UP = 0,
    ACTION_DOWN = 1,
    ACTION_LEFT = 2,
    ACTION_RIGHT = 3,
    ACTION_COUNT = 4
} Action;

typedef struct {
    int x, y;

    Network brain;

    int vision_radius;
    int vision_size;

    char **attr_names;
    float *attr_values;
    int num_attrs;

    int alive;
} Agent;

void agent_init(Agent *a, int x, int y, int vision_radius,
                int num_hidden_layers, int *neurons_per_layer,
                int num_attrs, char **attr_names);
void agent_read_vision(Agent *a, World *w, float *out);
void agent_build_input(Agent *a, World *w, float *input);
Action agent_decide(Agent *a, World *w);
void agent_move(Agent *a, World *w, Action act);
void agent_free(Agent *a);

#endif