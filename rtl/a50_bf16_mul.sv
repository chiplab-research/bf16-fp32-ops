// A50-BF16P2-001: BF16 x BF16 -> binary32 product, rounded once (RNE, gradual underflow).
// Bit-identical to sim/fp_reference/words.py mul32({a,16'h0}, {b,16'h0}). The 8x8 significand
// product is exact in 16 bits, so rounding happens only on binary32 range underflow (subnormal or
// zero results); overflow gives infinity. NaN in, or inf x 0, gives the canonical 0x7fc00000.
module a50_bf16_mul (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg  [31:0] p
);
    wire sign = a[15] ^ b[15];
    wire [7:0] ea = a[14:7], eb = b[14:7];
    wire a_nan = ea == 8'hFF && a[6:0] != 0, b_nan = eb == 8'hFF && b[6:0] != 0;
    wire a_inf = ea == 8'hFF && a[6:0] == 0, b_inf = eb == 8'hFF && b[6:0] == 0;
    wire a_zero = a[14:0] == 0, b_zero = b[14:0] == 0;
    wire [7:0] ma = {ea != 0, a[6:0]}, mb = {eb != 0, b[6:0]};
    wire [15:0] prod = ma * mb;
    // lsb exponent of a bf16 significand: (max(e,1) - 127 - 7); product: the sum of both
    wire signed [11:0] lsb_exp = $signed({4'd0, (ea == 0) ? 8'd1 : ea}) + $signed({4'd0, (eb == 0) ? 8'd1 : eb})
                                 - 12'sd268;
    wire [31:0] rounded;
    a50_round_pack rp(.sign(sign), .sig({12'd0, prod}), .lsb_exp(lsb_exp), .word(rounded));
    always @* begin
        if (a_nan || b_nan || (a_inf && b_zero) || (b_inf && a_zero))
            p = 32'h7FC00000;
        else if (a_inf || b_inf)
            p = {sign, 8'hFF, 23'd0};
        else if (a_zero || b_zero)
            p = {sign, 31'd0};
        else
            p = rounded;
    end
endmodule
