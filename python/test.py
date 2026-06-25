import numpy as np
import os

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "weights.txt")

INPUTS = 2
OUTPUTS = 1
NUM_LAYERS = 1
NEURONS_PER_LAYER = [3]
ACT_INPUT = "linear"
ACT_HIDDEN = "tanh"
ACT_OUTPUT = "sigmoid"

ACT_MAP = {
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(0, x),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
    "linear": lambda x: x,
    "softmax": lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))),
}

architecture = [INPUTS] + NEURONS_PER_LAYER + [OUTPUTS]

activations = []
for l in range(NUM_LAYERS + 1):
    if l == 0:
        activations.append(ACT_INPUT)
    elif l == NUM_LAYERS:
        activations.append(ACT_OUTPUT)
    else:
        activations.append(ACT_HIDDEN)

np.random.seed(42)
layers = []
prev = INPUTS
for l in range(NUM_LAYERS + 1):
    n = NEURONS_PER_LAYER[l] if l < NUM_LAYERS else OUTPUTS
    W = np.random.randn(n, prev)
    b = np.random.randn(n, 1)
    layers.append((W, b))
    prev = n

with open(WEIGHTS_FILE, "w") as f:
    for l in range(NUM_LAYERS + 1):
        W, b = layers[l]
        for n in range(W.shape[0]):
            for j in range(W.shape[1]):
                f.write(f"{W[n, j]}\n")
            f.write(f"{b[n, 0]}\n")

print(f"Weights saved to {WEIGHTS_FILE}")
print(f"Architecture: {architecture}")
print(f"Activations: {activations}")
print()

test_inputs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]

for inp in test_inputs:
    a = np.array(inp, dtype=float).reshape(-1, 1)
    for l in range(NUM_LAYERS + 1):
        W, b = layers[l]
        z = W @ a + b
        act_fn = ACT_MAP[activations[l]]
        a = act_fn(z)
    out = a.flatten()
    print(f"input {inp} -> output: {out.tolist()}")