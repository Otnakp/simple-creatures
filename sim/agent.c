#include "agent.h"
#include <stdlib.h>
#include <string.h>

static void init_brain(Agent *a, int num_hidden_layers, int *neurons_per_layer)
{
    a->brain.inputs = a->vision_size + a->num_attrs;
    a->brain.outputs = ACTION_COUNT;
    a->brain.numLayers = num_hidden_layers;
    a->brain.neuronsPerLayer = malloc(sizeof(int) * (num_hidden_layers > 0 ? num_hidden_layers : 1));
    for (int i = 0; i < num_hidden_layers; i++)
        a->brain.neuronsPerLayer[i] = neurons_per_layer[i];
    a->brain.actInput = ACT_LINEAR;
    a->brain.actHidden = ACT_TANH;
    a->brain.actOutput = ACT_LINEAR;
    network_init(&a->brain);
}

void agent_init(Agent *a, int x, int y, int vision_radius,
                int num_hidden_layers, int *neurons_per_layer,
                int num_attrs, char **attr_names)
{
    a->x = x;
    a->y = y;
    a->vision_radius = vision_radius;
    a->vision_size = (2 * vision_radius + 1) * (2 * vision_radius + 1);
    a->num_attrs = num_attrs;
    a->alive = 1;

    a->attr_names = malloc(sizeof(char *) * num_attrs);
    a->attr_values = malloc(sizeof(float) * num_attrs);
    for (int i = 0; i < num_attrs; i++) {
        a->attr_names[i] = strdup(attr_names[i]);
        a->attr_values[i] = 50.0f;
    }

    init_brain(a, num_hidden_layers, neurons_per_layer);
}

void agent_read_vision(Agent *a, World *w, float *out)
{
    int idx = 0;
    for (int dy = -a->vision_radius; dy <= a->vision_radius; dy++) {
        for (int dx = -a->vision_radius; dx <= a->vision_radius; dx++) {
            CellType c = world_get(w, a->x + dx, a->y + dy);
            switch (c) {
                case CELL_FOOD:   out[idx] = 1.0f; break;
                case CELL_POISON: out[idx] = -1.0f; break;
                default:          out[idx] = 0.0f; break;
            }
            idx++;
        }
    }
}

void agent_build_input(Agent *a, World *w, float *input)
{
    agent_read_vision(a, w, input);
    for (int i = 0; i < a->num_attrs; i++)
        input[a->vision_size + i] = a->attr_values[i] / 100.0f;
}

Action agent_decide(Agent *a, World *w)
{
    float *input = malloc(sizeof(float) * (a->vision_size + a->num_attrs));
    agent_build_input(a, w, input);

    float *out = feedforward(&a->brain, input);

    Action best = 0;
    float max_val = out[0];
    for (int i = 1; i < ACTION_COUNT; i++) {
        if (out[i] > max_val) {
            max_val = out[i];
            best = i;
        }
    }

    free(input);
    free(out);
    return best;
}

void agent_move(Agent *a, World *w, Action act)
{
    int nx = a->x, ny = a->y;
    switch (act) {
        case ACTION_UP:    ny--; break;
        case ACTION_DOWN:  ny++; break;
        case ACTION_LEFT:  nx--; break;
        case ACTION_RIGHT: nx++; break;
    }
    if (nx >= 0 && ny >= 0 && nx < w->width && ny < w->height) {
        a->x = nx;
        a->y = ny;
        CellType c = world_get(w, a->x, a->y);
        if (c == CELL_FOOD) {
            a->attr_values[0] = (a->attr_values[0] + 20.0f > 100.0f) ? 100.0f : a->attr_values[0] + 20.0f;
            world_set(w, a->x, a->y, CELL_EMPTY);
        } else if (c == CELL_POISON) {
            a->attr_values[0] -= 30.0f;
            world_set(w, a->x, a->y, CELL_EMPTY);
        }
    }
    a->attr_values[1] += 1.0f;
    if (a->attr_values[1] > 100.0f) a->attr_values[1] = 100.0f;
    a->attr_values[0] -= 0.5f;
    if (a->attr_values[0] <= 0.0f) {
        a->attr_values[0] = 0.0f;
        a->alive = 0;
    }
}

void agent_free(Agent *a)
{
    network_free(&a->brain);
    free(a->brain.neuronsPerLayer);
    for (int i = 0; i < a->num_attrs; i++)
        free(a->attr_names[i]);
    free(a->attr_names);
    free(a->attr_values);
}