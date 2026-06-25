#include "../network/network.h"
#include "world.h"
#include "agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MUTATION_RATE 0.1f
#define MUTATION_MAG 0.2f

static void mutate(Agent *dst, Agent *src)
{
    for (int l = 0; l < src->total_layers; l++) {
        int postSize = src->post_sizes[l];
        int preSize = src->pre_sizes[l];
        for (int j = 0; j < postSize; j++) {
            for (int i = 0; i < preSize; i++) {
                float w = src->brain.weights[l][j][i];
                if ((float)rand() / RAND_MAX < MUTATION_RATE)
                    w += (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * MUTATION_MAG;
                dst->brain.weights[l][j][i] = w;
            }
        }
    }
}

int main(void)
{
    srand(time(NULL));

    int width = 30, height = 18;
    int num_agents = 50;
    int max_generations = 10000;
    int max_steps = 2000;

    World world;
    world_init(&world, width, height);
    world_populate(&world, 100, 30);

    char *attr_names[] = { "Vita", "Fame" };
    int neurons[] = { 16 };
    Agent agents[50];
    for (int i = 0; i < num_agents; i++) {
        int ax = 2 + rand() % (width - 4);
        int ay = 2 + rand() % (height - 4);
        agent_init(&agents[i], ax, ay, 2, 1, neurons, 2, attr_names);
    }

    int gen = 0;
    while (gen < max_generations) {
        int steps = 0;
        int alive_count = num_agents;

        while (alive_count > 0 && steps < max_steps) {
            for (int i = 0; i < num_agents; i++) {
                if (!agents[i].alive) continue;
                Action act = agent_decide(&agents[i], &world);
                agent_move(&agents[i], &world, act);
            }

            alive_count = 0;
            for (int i = 0; i < num_agents; i++)
                if (agents[i].alive) alive_count++;

            steps++;
        }

        if (gen % 50 == 0 || gen < 5) {
            float best_score = -999;
            int best_idx = 0;
            for (int i = 0; i < num_agents; i++) {
                float s = agents[i].food_eaten - agents[i].poison_eaten;
                if (s > best_score) { best_score = s; best_idx = i; }
            }
            printf("Gen %4d: steps=%4d alive=%2d/%d | best=agent%d score=%.0f (food=%.0f poison=%.0f learn=%d)\n",
                   gen, steps, alive_count, num_agents,
                   best_idx, best_score,
                   agents[best_idx].food_eaten, agents[best_idx].poison_eaten,
                   agents[best_idx].learn_events);
        }

        int rank[50];
        for (int i = 0; i < num_agents; i++) rank[i] = i;
        for (int i = 0; i < num_agents - 1; i++) {
            for (int j = i + 1; j < num_agents; j++) {
                float si = agents[rank[i]].food_eaten - agents[rank[i]].poison_eaten;
                float sj = agents[rank[j]].food_eaten - agents[rank[j]].poison_eaten;
                if (sj > si) {
                    int tmp = rank[i]; rank[i] = rank[j]; rank[j] = tmp;
                }
            }
        }

        int keep = num_agents / 2;
        for (int k = 0; k < keep; k++) {
            Agent *src = &agents[rank[k]];
            int ax = 2 + rand() % (width - 4);
            int ay = 2 + rand() % (height - 4);
            agent_reset(src, ax, ay);
        }
        for (int k = keep; k < num_agents; k++) {
            Agent *src = &agents[rank[k % keep]];
            Agent *dst = &agents[rank[k]];
            mutate(dst, src);
            int ax = 2 + rand() % (width - 4);
            int ay = 2 + rand() % (height - 4);
            agent_reset(dst, ax, ay);
        }

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                world_set(&world, x, y, CELL_EMPTY);
world_populate(&world, 100, 30);

        gen++;
    }

    printf("=== Final gen %d ===\n", gen);
    for (int i = 0; i < num_agents; i++) {
        printf("Agent %2d: food=%.0f poison=%.0f score=%g learn=%d\n",
               i, agents[i].food_eaten, agents[i].poison_eaten,
               agents[i].food_eaten - agents[i].poison_eaten,
               agents[i].learn_events);
    }

    for (int i = 0; i < num_agents; i++)
        agent_free(&agents[i]);
    world_free(&world);
    return 0;
}