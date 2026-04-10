module xor_net (
    input  [1:0] sw,      // Physical FPGA switches
    output [0:0] led      // Physical FPGA LED
);
    // Hidden layer weights and biases (Q1.3) [cite: 65-68]
    localparam signed [7:0] W1_00 = 8'shCA; // -54
    localparam signed [7:0] W1_01 = 8'sh29; //  41
    localparam signed [7:0] W1_10 = 8'shCA; // -54
    localparam signed [7:0] W1_11 = 8'sh29; //  41
    localparam signed [7:0] B1_0  = 8'sh16; //  22
    localparam signed [7:0] B1_1  = 8'shC0; // -64

    // Output layer weights and bias (Q1.3) [cite: 69-72]
    localparam signed [7:0] W2_0  = 8'shA9; // -87
    localparam signed [7:0] W2_1  = 8'shA7; // -89
    localparam signed [7:0] B2    = 8'sh2B; //  43

    // Internal wires for layer connections
    wire signed [7:0] h0_out, h1_out;
    wire signed [7:0] final_out;

    // Convert binary switch inputs (0/1) to Q1.3 format (0/8) [cite: 33, 45]
    wire signed [7:0] x0 = (sw[0]) ? 8'sd8 : 8'sd0;
    wire signed [7:0] x1 = (sw[1]) ? 8'sd8 : 8'sd0;

    // --- Layer 1: Hidden Layer (2 Neurons) [cite: 41, 61] ---
    // Hidden Neuron 0
    neuron n_h0 ( .x0(x0), .x1(x1), .w0(W1_00), .w1(W1_10), .b(B1_0), .out(h0_out) );
    
    // Hidden Neuron 1
    neuron n_h1 ( .x0(x0), .x1(x1), .w0(W1_01), .w1(W1_11), .b(B1_1), .out(h1_out) );

    // --- Layer 2: Output Layer (1 Neuron) [cite: 42, 61] ---
    neuron n_out ( .x0(h0_out), .x1(h1_out), .w0(W2_0), .w1(W2_1), .b(B2), .out(final_out) );

    // Map final activated output to the LED [cite: 62]
    assign led[0] = (final_out > 0);

endmodule
