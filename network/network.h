#ifndef NETWORK_H
#define NETWORK_H

#include <stdint.h>

typedef enum
{
    ACT_SIGMOID,
    ACT_TANH,
    ACT_RELU,
    ACT_LEAKY_RELU,
    ACT_LINEAR,
    ACT_SOFTMAX
} Activation;

typedef struct
{
    int inputs;
    int outputs;
    int numLayers;
    int *neuronsPerLayer;

    Activation actHidden;
    Activation actInput;
    Activation actOutput;

    float ***weights;

    float **pre_act;
    float **post_act;
    int acts_allocated;
} Network;

void network_init(Network *net);
void network_load_weights(Network *net, const char *filename);
void network_free(Network *net);
float *feedforward(Network *net, const float *input);

#endif // NETWORK_H