`timescale 1ns / 1ps

module tb_xor_net();
    // Inputs to the network
    reg [1:0] sw;
    // Output from the network
    wire [0:0] led;

    // Instantiate the Unit Under Test (UUT)
    xor_net uut (
        .sw(sw),
        .led(led)
    );

    initial begin
        // Initialize Inputs
        $display("Starting XOR Neural Network Simulation...");
        $display("---------------------------------------");
        $display("SW1  SW0  |  LED (XOR Result)");
        $display("---------------------------------------");

        // Test Case 1: 0 ^ 0 = 0
        sw = 2'b00; #10;
        $display(" 0    0   |   %b", led[0]);

        // Test Case 2: 0 ^ 1 = 1
        sw = 2'b01; #10;
        $display(" 0    1   |   %b", led[0]);

        // Test Case 3: 1 ^ 0 = 1
        sw = 2'b10; #10;
        $display(" 1    0   |   %b", led[0]);

        // Test Case 4: 1 ^ 1 = 0
        sw = 2'b11; #10;
        $display(" 1    1   |   %b", led[0]);

        $display("---------------------------------------");
        $finish;
    end
endmodule
