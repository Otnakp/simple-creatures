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
    net->acts_allocated = 0;

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

float *feedforward(Network *net, const float *input)
{
    int totalLayers = net->numLayers + 1;

    if (!net->acts_allocated) {
        net->pre_act = malloc(totalLayers * sizeof(float *));
        net->post_act = malloc(totalLayers * sizeof(float *));
        int prev = net->inputs;
        for (int l = 0; l < totalLayers; l++) {
            int layerSize = (l < net->numLayers) ? net->neuronsPerLayer[l] : net->outputs;
            net->pre_act[l] = malloc(sizeof(float) * (prev + 1));
            net->post_act[l] = malloc(sizeof(float) * layerSize);
            prev = layerSize;
        }
        net->acts_allocated = 1;
    }

    int prevSize = net->inputs;
    for (int i = 0; i < prevSize; i++)
        net->pre_act[0][i] = input[i];
    net->pre_act[0][prevSize] = 1.0f;

    float *cur = net->pre_act[0];

    for (int l = 0; l < totalLayers; l++) {
        int layerSize = (l < net->numLayers) ? net->neuronsPerLayer[l] : net->outputs;
        Activation act = (l == 0) ? net->actInput
                       : (l == net->numLayers) ? net->actOutput
                       : net->actHidden;
        for (int i = 0; i < layerSize; i++) {
            float x = net->weights[l][i][prevSize];
            for (int j = 0; j < prevSize; j++)
                x += cur[j] * net->weights[l][i][j];
            net->post_act[l][i] = (act == ACT_SOFTMAX) ? x : activate(x, act);
        }
        if (act == ACT_SOFTMAX)
            apply_softmax(net->post_act[l], layerSize);

        if (l < totalLayers - 1) {
            for (int i = 0; i < layerSize; i++)
                net->pre_act[l+1][i] = net->post_act[l][i];
            net->pre_act[l+1][layerSize] = 1.0f;
            cur = net->pre_act[l+1];
        }
        prevSize = layerSize;
    }

    float *out = malloc(sizeof(float) * net->outputs);
    for (int i = 0; i < net->outputs; i++)
        out[i] = net->post_act[totalLayers-1][i];
    return out;
}

void network_load_weights(Network *net, const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f) return;

    int totalLayers = net->numLayers + 1;
    net->weights = malloc(totalLayers * sizeof(float **));
    net->acts_allocated = 0;

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
        if (net->acts_allocated) {
            free(net->pre_act[l]);
            free(net->post_act[l]);
        }
    }
    free(net->weights);
    if (net->acts_allocated) {
        free(net->pre_act);
        free(net->post_act);
    }
}