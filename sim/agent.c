#include "agent.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TRACE_CAP 15
#define ACTIVE_THRESHOLD 0.3f
#define LEARNING_RATE 0.15f
#define DECAY_K 0.15f

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

    a->total_layers = a->brain.numLayers + 1;
    a->pre_sizes = malloc(sizeof(int) * a->total_layers);
    a->post_sizes = malloc(sizeof(int) * a->total_layers);
    int prev = a->brain.inputs;
    for (int l = 0; l < a->total_layers; l++) {
        int layerSize = (l < a->brain.numLayers) ? a->brain.neuronsPerLayer[l] : a->brain.outputs;
        a->pre_sizes[l] = prev + 1;
        a->post_sizes[l] = layerSize;
        prev = layerSize;
    }

    a->trace_cap = TRACE_CAP;
    a->trace_head = 0;
    a->trace_count = 0;
    a->trace_pre = malloc(sizeof(float **) * TRACE_CAP);
    a->trace_post = malloc(sizeof(float **) * TRACE_CAP);
    a->trace_actions = malloc(sizeof(Action) * TRACE_CAP);
    for (int t = 0; t < TRACE_CAP; t++) {
        a->trace_pre[t] = malloc(sizeof(float *) * a->total_layers);
        a->trace_post[t] = malloc(sizeof(float *) * a->total_layers);
        for (int l = 0; l < a->total_layers; l++) {
            a->trace_pre[t][l] = malloc(sizeof(float) * a->pre_sizes[l]);
            a->trace_post[t][l] = malloc(sizeof(float) * a->post_sizes[l]);
        }
    }
    a->food_eaten = 0;
    a->poison_eaten = 0;
    a->learn_events = 0;
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

    int t = a->trace_head;
    for (int l = 0; l < a->total_layers; l++) {
        for (int i = 0; i < a->pre_sizes[l]; i++)
            a->trace_pre[t][l][i] = a->brain.pre_act[l][i];
        for (int i = 0; i < a->post_sizes[l]; i++)
            a->trace_post[t][l][i] = a->brain.post_act[l][i];
    }
    a->trace_actions[t] = best;
    a->trace_head = (a->trace_head + 1) % a->trace_cap;
    if (a->trace_count < a->trace_cap) a->trace_count++;

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
            a->food_eaten += 1;
            a->attr_values[0] = (a->attr_values[0] + 20.0f > 100.0f) ? 100.0f : a->attr_values[0] + 20.0f;
            world_set(w, a->x, a->y, CELL_EMPTY);
        } else if (c == CELL_POISON) {
            a->poison_eaten += 1;
            a->attr_values[0] -= 30.0f;
            world_set(w, a->x, a->y, CELL_EMPTY);
        }
    }
    a->attr_values[1] += 1.0f;
    if (a->attr_values[1] > 100.0f) a->attr_values[1] = 100.0f;
    a->attr_values[0] -= 0.1f;
    if (a->attr_values[0] <= 0.0f) {
        a->attr_values[0] = 0.0f;
        a->alive = 0;
    }
}

void agent_learn(Agent *a, float reward)
{
    a->learn_events++;
    int total = a->total_layers;
    int n = a->trace_count;

    for (int s = 0; s < n; s++) {
        int idx = (a->trace_head - 1 - s + a->trace_cap * 10) % a->trace_cap;
        float decay = expf(-DECAY_K * s);

        for (int l = 0; l < total; l++) {
            int postSize = a->post_sizes[l];
            int preSize = a->pre_sizes[l];
            int is_output = (l == total - 1);
            Action chosen = a->trace_actions[idx];

            for (int j = 0; j < postSize; j++) {
                if (is_output && j != chosen) continue;
                float post_act = a->trace_post[idx][l][j];
                if (post_act < ACTIVE_THRESHOLD && post_act > -ACTIVE_THRESHOLD) continue;

                for (int i = 0; i < preSize; i++) {
                    float pre_act = a->trace_pre[idx][l][i];
                    if (pre_act < ACTIVE_THRESHOLD && pre_act > -ACTIVE_THRESHOLD) continue;
                    float w = a->brain.weights[l][j][i];
                    float contrib = pre_act * w;
                    float sgn = (contrib >= 0) ? 1.0f : -1.0f;
                    a->brain.weights[l][j][i] += LEARNING_RATE * reward * decay * sgn;
                }
            }
        }
    }
}

void agent_reset(Agent *a, int x, int y)
{
    a->x = x;
    a->y = y;
    a->alive = 1;
    a->attr_values[0] = 50.0f;
    a->attr_values[1] = 0.0f;
    a->trace_count = 0;
    a->trace_head = 0;
    a->food_eaten = 0;
    a->poison_eaten = 0;
    a->learn_events = 0;
}

void agent_free(Agent *a)
{
    network_free(&a->brain);
    free(a->brain.neuronsPerLayer);
    for (int i = 0; i < a->num_attrs; i++)
        free(a->attr_names[i]);
    free(a->attr_names);
    free(a->attr_values);

    for (int t = 0; t < a->trace_cap; t++) {
        for (int l = 0; l < a->total_layers; l++) {
            free(a->trace_pre[t][l]);
            free(a->trace_post[t][l]);
        }
        free(a->trace_pre[t]);
        free(a->trace_post[t]);
    }
    free(a->trace_pre);
    free(a->trace_post);
    free(a->trace_actions);
    free(a->pre_sizes);
    free(a->post_sizes);
}