// A50-BF16P2-001: first half of a binary32 adder (specials, swap, align, add). Its outputs are
// registered and then rounded by a50_round_pack, so one add takes two pipeline stages.
// The smaller operand is shifted right with three extra bits and a jammed sticky bit, so sum is
// exact or lies strictly between the same two even integers as the exact sum. Every rounding
// boundary a50_round_pack can use is even (see the PR notes), so the single rounding equals
// words.py add32 (RNE; exact cancellation gives +0, only -0 + -0 gives -0; inf - inf is NaN).
module a50_add32_align (
    input  wire [31:0]        a,
    input  wire [31:0]        b,
    output reg                special,      // result is special_word, not a rounded sum
    output reg  [31:0]        special_word,
    output reg                sign,
    output reg  [27:0]        sum,
    output reg  signed [11:0] lsb_exp
);
    wire a_nan = a[30:23] == 8'hFF && a[22:0] != 0, b_nan = b[30:23] == 8'hFF && b[22:0] != 0;
    wire a_inf = a[30:23] == 8'hFF && a[22:0] == 0, b_inf = b[30:23] == 8'hFF && b[22:0] == 0;
    wire swap = b[30:0] > a[30:0];
    wire [31:0] hi = swap ? b : a, lo = swap ? a : b;
    wire [7:0] eb = (hi[30:23] == 0) ? 8'd1 : hi[30:23];
    wire [7:0] es = (lo[30:23] == 0) ? 8'd1 : lo[30:23];
    wire [26:0] mb = {hi[30:23] != 0, hi[22:0], 3'b000};
    wire [26:0] ms = {lo[30:23] != 0, lo[22:0], 3'b000};
    wire [7:0] d_full = eb - es;
    wire [4:0] d = (d_full > 8'd27) ? 5'd27 : d_full[4:0];
    wire [26:0] shifted = ms >> d;
    wire sticky = (ms & ((27'd1 << d) - 27'd1)) != 0;
    wire [26:0] sm = shifted | {26'd0, sticky};
    wire [27:0] s = (hi[31] == lo[31]) ? {1'b0, mb} + {1'b0, sm} : {1'b0, mb} - {1'b0, sm};
    always @* begin
        special = 1'b0;
        special_word = 32'd0;
        if (a_nan || b_nan || (a_inf && b_inf && (a[31] ^ b[31]))) begin
            special = 1'b1; special_word = 32'h7FC00000;
        end else if (a_inf) begin
            special = 1'b1; special_word = a;
        end else if (b_inf) begin
            special = 1'b1; special_word = b;
        end
        sum = s;
        lsb_exp = $signed({4'd0, eb}) - 12'sd153;
        sign = (s == 0) ? (a[31] & b[31]) : hi[31];
    end
endmodule
