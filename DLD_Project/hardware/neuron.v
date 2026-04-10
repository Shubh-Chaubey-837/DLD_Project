// neuron.v - Q1.3 Fixed-Point Neuron
module neuron (
    input  signed [7:0] x0, x1,     // Q1.3 Inputs
    input  signed [7:0] w0, w1,     // Q1.3 Weights
    input  signed [7:0] b,          // Q1.3 Bias
    output signed [7:0] out         // Q1.3 Activated Output
);
    // Intermediate products (8-bit * 8-bit = 16-bit)
    wire signed [15:0] prod0 = x0 * w0;
    wire signed [15:0] prod1 = x1 * w1;
    
    // Summation logic: 
    // Shift products right by 3 to return to Q1.3 format before adding the bias [cite: 65]
    wire signed [15:0] sum = (prod0 >>> 3) + (prod1 >>> 3) + b;

    // Activation Function: STEP [cite: 62, 73]
    // If sum is > 0, output is 1.0 (which is 8 in Q1.3)
    // If sum is <= 0, output is 0.0
    assign out = (sum > 0) ? 8'sd8 : 8'sd0; 

endmodule
