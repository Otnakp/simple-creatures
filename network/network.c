#include "network.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

static float random_weight(void)
{
    return ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

void network_init(Network *net)
{
    int totalLayers = net->numLayers + 1;
    net->weights = malloc(totalLayers * sizeof(float **));

    int prevSize = net->inputs;

    for (int l = 0; l < net->numLayers; l++)
    {
        int layerSize = net->neuronsPerLayer[l];
        net->weights[l] = malloc(layerSize * sizeof(float *));

        for (int n = 0; n < layerSize; n++)
        {
            net->weights[l][n] = malloc((prevSize + 1) * sizeof(float));
            for (int w = 0; w < prevSize + 1; w++)
                net->weights[l][n][w] = random_weight();
        }
        prevSize = layerSize;
    }

    net->weights[net->numLayers] = malloc(net->outputs * sizeof(float *));
    for (int n = 0; n < net->outputs; n++)
    {
        net->weights[net->numLayers][n] = malloc((prevSize + 1) * sizeof(float));
        for (int w = 0; w < prevSize + 1; w++)
            net->weights[net->numLayers][n][w] = random_weight();
    }
}

static float activate(float x, Activation a)
{
    switch (a) {
        case ACT_SIGMOID:    return 1.0f / (1.0f + expf(-x));
        case ACT_TANH:       return tanhf(x);
        case ACT_RELU:       return x > 0 ? x : 0.0f;
        case ACT_LEAKY_RELU: return x > 0 ? x : 0.01f * x;
        case ACT_LINEAR:     return x;
        default:             return x;
    }
}

static void apply_softmax(float *v, int n)
{
    float max = v[0];
    for (int i = 1; i < n; i++)
        if (v[i] > max) max = v[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        v[i] = expf(v[i] - max);
        sum += v[i];
    }
    for (int i = 0; i < n; i++)
        v[i] /= sum;
}

static float *vec_mat_mul(float *v, int h, float **m, int w, Activation act)
{
    float *r = malloc(sizeof(float) * w);
    for (int i = 0; i < w; i++) {
        float x = m[i][h];
        for (int j = 0; j < h; j++)
            x += v[j] * m[i][j];
        r[i] = (act == ACT_SOFTMAX) ? x : activate(x, act);
    }
    if (act == ACT_SOFTMAX)
        apply_softmax(r, w);
    return r;
}

float *feedforward(const Network *net, const float *input)
{
    int totalLayers = net->numLayers + 1;
    int prevSize = net->inputs;
    float *cur = malloc(sizeof(float) * (prevSize + 1));
    for (int i = 0; i < prevSize; i++)
        cur[i] = input[i];
    cur[prevSize] = 1.0f;

    for (int l = 0; l < totalLayers; l++) {
        int layerSize = (l < net->numLayers) ? net->neuronsPerLayer[l] : net->outputs;
        Activation act = (l == 0) ? net->actInput
                       : (l == net->numLayers) ? net->actOutput
                       : net->actHidden;
        float *next = vec_mat_mul(cur, prevSize, net->weights[l], layerSize, act);
        free(cur);
        if (l == totalLayers - 1) {
            cur = next;
        } else {
            cur = malloc(sizeof(float) * (layerSize + 1));
            for (int i = 0; i < layerSize; i++)
                cur[i] = next[i];
            cur[layerSize] = 1.0f;
            free(next);
        }
        prevSize = layerSize;
    }
    return cur;
}

void network_load_weights(Network *net, const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f) return;

    int totalLayers = net->numLayers + 1;
    net->weights = malloc(totalLayers * sizeof(float **));

    int prevSize = net->inputs;
    for (int l = 0; l < totalLayers; l++) {
        int layerSize = (l < net->numLayers) ? net->neuronsPerLayer[l] : net->outputs;
        net->weights[l] = malloc(layerSize * sizeof(float *));
        for (int n = 0; n < layerSize; n++) {
            net->weights[l][n] = malloc((prevSize + 1) * sizeof(float));
            for (int w = 0; w < prevSize + 1; w++)
                fscanf(f, "%f", &net->weights[l][n][w]);
        }
        prevSize = layerSize;
    }
    fclose(f);
}

void network_free(Network *net)
{
    int totalLayers = net->numLayers + 1;
    for (int l = 0; l < totalLayers; l++)
    {
        int layerSize = (l < net->numLayers) ? net->neuronsPerLayer[l] : net->outputs;
        for (int n = 0; n < layerSize; n++)
            free(net->weights[l][n]);
        free(net->weights[l]);
    }
    free(net->weights);
}