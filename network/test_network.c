#include "network.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    Network net;
    net.inputs = 2;
    net.outputs = 1;
    net.numLayers = 1;
    net.neuronsPerLayer = malloc(sizeof(int) * 1);
    net.neuronsPerLayer[0] = 3;
    net.actInput = ACT_LINEAR;
    net.actHidden = ACT_TANH;
    net.actOutput = ACT_SIGMOID;

    network_load_weights(&net, "../python/weights.txt");

    float test_inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    for (int t = 0; t < 4; t++) {
        float *out = feedforward(&net, test_inputs[t]);
        printf("input [%.0f, %.0f] -> output: ", test_inputs[t][0], test_inputs[t][1]);
        for (int i = 0; i < net.outputs; i++)
            printf("%f ", out[i]);
        printf("\n");
        free(out);
    }

    network_free(&net);
    free(net.neuronsPerLayer);
    return 0;
}