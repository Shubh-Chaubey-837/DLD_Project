// ==========================================================
// XOR 2-2-1 MLP  –  Fixed-Point Weights
// Hidden activation : Sigmoid (approx. LUT in hardware)
// Output activation : STEP
// Format            : Q1.3  (n=3, S=8)
// After every MAC   : right-shift the accumulator by 3 bits
// ==========================================================

// ── Hidden layer weights ──
localparam signed [7:0] W1_00 = 8'shCA;  // x0 -> hidden-0
localparam signed [7:0] W1_01 = 8'sh29;  // x0 -> hidden-1
localparam signed [7:0] W1_10 = 8'shCA;  // x1 -> hidden-0
localparam signed [7:0] W1_11 = 8'sh29;  // x1 -> hidden-1

// ── Hidden layer biases ──
localparam signed [7:0] B1_0  = 8'sh16;    // bias -> hidden-0
localparam signed [7:0] B1_1  = 8'shC0;    // bias -> hidden-1

// ── Output layer weights ──
localparam signed [7:0] W2_0  = 8'shA9;    // hidden-0 -> y
localparam signed [7:0] W2_1  = 8'shA7;    // hidden-1 -> y

// ── Output layer bias ──
localparam signed [7:0] B2    = 8'sh2B;        // bias -> y
