"""
xor_mlp_train.py
================
Trains a 2-2-1 MLP on the XOR problem using NumPy only (no ML libraries).
After training, weights and biases are quantised to 8-bit signed integers
using the Fixed-Point scheme from the project document:

    W_int = round(W_float * 2^n)     where S = 2^n

The fractional-bit count n is chosen automatically so that no weight
saturates the 8-bit signed range [-128 .. 127].  You can also set
FRAC_BITS manually to override auto-selection.

Outputs
-------
Console          – training log, inference truth-table, quantisation table
                   and a ready-to-paste Verilog localparam block
weights.mem      – plain hex, one byte per line  (use with $readmemh)
weights.txt      – signed decimal, one per line  (for easy inspection)
verilog_params.v – localparam block, ready to paste into Verilog

Activation choices
------------------
Hidden layer : Sigmoid  (smooth gradient  → good backprop)
Output layer : Step OR ReLU  (hardware-friendly; set OUTPUT_ACT below)
"""

import numpy as np

# ───────────────────────── Hyper-parameters ──────────────────────────────
SEED         = 1            # fixed seed for reproducibility
LR           = 0.5          # learning rate
EPOCHS       = 30_000       # training iterations
OUTPUT_ACT   = "step"       # "step"  or  "relu"
FRAC_BITS    = None         # None = auto-select; or set e.g. FRAC_BITS = 3
PRINT_EVERY  = 10_000
# ─────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)

# ── XOR truth table ──
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float64)          # (4, 2)
Y = np.array([[0], [1], [1], [0]], dtype=np.float64)   # (4, 1)

# ── Weight initialisation (small uniform – keeps converged weights compact) ──
W1 = np.random.uniform(-0.5, 0.5, (2, 2))    # input  → hidden   (2 in, 2 out)
b1 = np.zeros((1, 2))                          # hidden biases
W2 = np.random.uniform(-0.5, 0.5, (2, 1))    # hidden → output   (2 in, 1 out)
b2 = np.zeros((1, 1))                          # output bias

# ── Activation helpers ──
def sigmoid(z):    return 1.0 / (1.0 + np.exp(-z))
def sigmoid_d(a):  return a * (1.0 - a)         # derivative in terms of activation
def step_fn(z):    return (z >= 0.5).astype(float)
def relu_fn(z):    return np.maximum(0.0, z)

def output_act(z):
    return step_fn(z) if OUTPUT_ACT == "step" else relu_fn(z)

# ── Forward pass (sigmoid output kept for training; hardware uses output_act) ──
def forward(x):
    z1 = x @ W1 + b1;  a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2;  a2 = sigmoid(z2)
    return z1, a1, z2, a2

# ──────────────────────────── Training loop ──────────────────────────────
SEP = "=" * 62
print(SEP)
print("  2-2-1 XOR MLP  –  NumPy Backpropagation")
print(f"  LR={LR}  Epochs={EPOCHS}  Output activation: {OUTPUT_ACT.upper()}")
print(SEP)

for epoch in range(1, EPOCHS + 1):
    z1, a1, z2, a2 = forward(X)
    loss = float(np.mean((Y - a2) ** 2))

    # ── Backpropagation ──
    delta2 = (a2 - Y) * sigmoid_d(a2)              # (4,1)
    dW2 = a1.T @ delta2                              # (2,1)
    db2 = delta2.sum(axis=0, keepdims=True)          # (1,1)

    delta1 = (delta2 @ W2.T) * sigmoid_d(a1)        # (4,2)
    dW1 = X.T @ delta1                               # (2,2)
    db1 = delta1.sum(axis=0, keepdims=True)          # (1,2)

    W2 -= LR * dW2;  b2 -= LR * db2
    W1 -= LR * dW1;  b1 -= LR * db1

    if epoch == 1 or epoch % PRINT_EVERY == 0:
        print(f"  Epoch {epoch:>6d}   MSE = {loss:.8f}")

print()

# ─────────────────────── Inference verification ──────────────────────────
_, a1_ev, z2_ev, _ = forward(X)
a2_hw = output_act(z2_ev)

print("── Inference with hardware activation ──")
print(f"  {'X1':>3} {'X2':>3}  {'Target':>7}  {'Output':>7}  {'OK?':>5}")
correct = 0
for i in range(4):
    pred = int(a2_hw[i, 0] >= 0.5) if OUTPUT_ACT == "relu" else int(a2_hw[i, 0])
    tgt  = int(Y[i, 0])
    ok   = "OK" if pred == tgt else "FAIL"
    correct += (pred == tgt)
    print(f"   {int(X[i,0])}   {int(X[i,1])}   {tgt:>6}   {pred:>6}   {ok}")
print(f"\n  Accuracy : {correct}/4  {'PERFECT' if correct == 4 else 'Check training'}")
print()

# ──────────────────── Auto-select fractional bits ────────────────────────
all_fp = np.concatenate([W1.flatten(), b1.flatten(),
                          W2.flatten(), b2.flatten()])
w_max = float(np.max(np.abs(all_fp)))

if FRAC_BITS is None:
    # Largest n such that w_max * 2^n <= 127 (no saturation)
    n = 0
    while w_max * (2 ** (n + 1)) <= 127.0:
        n += 1
    FRAC_BITS = n

S = 2 ** FRAC_BITS
print(f"  Max |weight| in network : {w_max:.5f}")
print(f"  Selected n = {FRAC_BITS}  →  S = {S}")
print(f"  Max scaled integer      : {w_max * S:.2f}  (limit = 127)")
print()

# ──────────────────── Quantisation ───────────────────────────────────────
def quantise(fp):
    """Round float to nearest 8-bit signed int; clamp to [-128, 127]."""
    return int(max(-128, min(127, round(fp * S))))

def hex8(v):
    """Two's-complement hex, always 2 uppercase digits."""
    return format(v & 0xFF, '02X')

# Logical order follows hardware feed-forward path
params = [
    ("W1[0,0]", W1[0, 0]),   # x0  → hidden-0
    ("W1[0,1]", W1[0, 1]),   # x0  → hidden-1
    ("W1[1,0]", W1[1, 0]),   # x1  → hidden-0
    ("W1[1,1]", W1[1, 1]),   # x1  → hidden-1
    ("b1[0]",   b1[0, 0]),   # bias → hidden-0
    ("b1[1]",   b1[0, 1]),   # bias → hidden-1
    ("W2[0]",   W2[0, 0]),   # hidden-0 → output
    ("W2[1]",   W2[1, 0]),   # hidden-1 → output
    ("b2",      b2[0, 0]),   # bias → output
]
params_q = [(name, quantise(fp)) for name, fp in params]
d = dict(params_q)

print(SEP)
print(f"  Fixed-Point Quantisation   n = {FRAC_BITS}   S = {S}")
print(SEP)
print(f"  {'Parameter':<12} {'Float':>10}  {'W_int':>8}  {'Hex':>6}")
print("  " + "-" * 44)
for (name, fp), (_, iq) in zip(params, params_q):
    sat = "  <- SATURATED" if abs(iq) == 128 else ""
    print(f"  {name:<12} {fp:>10.5f}  {iq:>8}     {hex8(iq)}{sat}")
print()

# ─────────────────── Verilog localparam block ────────────────────────────
vlog = (
    "// " + "=" * 58 + "\n"
    f"// XOR 2-2-1 MLP  –  Fixed-Point Weights\n"
    f"// Hidden activation : Sigmoid (approx. LUT in hardware)\n"
    f"// Output activation : {OUTPUT_ACT.upper()}\n"
    f"// Format            : Q1.{FRAC_BITS}  (n={FRAC_BITS}, S={S})\n"
    f"// After every MAC   : right-shift the accumulator by {FRAC_BITS} bits\n"
    "// " + "=" * 58 + "\n\n"
    "// ── Hidden layer weights ──\n"
    f"localparam signed [7:0] W1_00 = 8'sh{hex8(d['W1[0,0]'])};  // x0 -> hidden-0\n"
    f"localparam signed [7:0] W1_01 = 8'sh{hex8(d['W1[0,1]'])};  // x0 -> hidden-1\n"
    f"localparam signed [7:0] W1_10 = 8'sh{hex8(d['W1[1,0]'])};  // x1 -> hidden-0\n"
    f"localparam signed [7:0] W1_11 = 8'sh{hex8(d['W1[1,1]'])};  // x1 -> hidden-1\n\n"
    "// ── Hidden layer biases ──\n"
    f"localparam signed [7:0] B1_0  = 8'sh{hex8(d['b1[0]'])};    // bias -> hidden-0\n"
    f"localparam signed [7:0] B1_1  = 8'sh{hex8(d['b1[1]'])};    // bias -> hidden-1\n\n"
    "// ── Output layer weights ──\n"
    f"localparam signed [7:0] W2_0  = 8'sh{hex8(d['W2[0]'])};    // hidden-0 -> y\n"
    f"localparam signed [7:0] W2_1  = 8'sh{hex8(d['W2[1]'])};    // hidden-1 -> y\n\n"
    "// ── Output layer bias ──\n"
    f"localparam signed [7:0] B2    = 8'sh{hex8(d['b2'])};        // bias -> y\n"
)

print(SEP)
print("  Verilog localparam block")
print(SEP)
print(vlog)

# ───────────────────────── Write output files ─────────────────────────────
out = "/mnt/user-data/outputs"

# 1. weights.mem  (for $readmemh)
with open(f"{out}/weights.mem", "w") as f:
    f.write("// XOR MLP weights – load with $readmemh(\"weights.mem\", mem)\n")
    f.write(f"// n={FRAC_BITS}  S={S}  activation={OUTPUT_ACT}\n")
    f.write("// Order: W1_00 W1_01 W1_10 W1_11  b1_0 b1_1  W2_0 W2_1  b2\n")
    for i, (name, iq) in enumerate(params_q):
        f.write(f"{hex8(iq)}  // [{i}] {name}\n")
print("  Written: weights.mem")

# 2. weights.txt  (signed decimal)
with open(f"{out}/weights.txt", "w") as f:
    f.write(f"# XOR MLP fixed-point weights  n={FRAC_BITS}  S={S}  output={OUTPUT_ACT}\n")
    for name, iq in params_q:
        f.write(f"{iq:4d}   # {name}\n")
print("  Written: weights.txt")

# 3. verilog_params.v
with open(f"{out}/verilog_params.v", "w") as f:
    f.write(vlog)
print("  Written: verilog_params.v")

print()
print("  Paste verilog_params.v into your Verilog top module, OR")
print("  use  $readmemh(\"weights.mem\", mem_array)  to load a ROM/RAM.")
print()
