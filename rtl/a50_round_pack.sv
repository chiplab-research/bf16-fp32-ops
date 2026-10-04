// A50-BF16P2-001: round an exact (or sticky-jammed) magnitude once to binary32, RNE, gradual underflow.
// value = sig * 2^lsb_exp. Mirrors sim/fp_reference/words.py _pack(): quantum = max(top_exp - 23, -149),
// one RNE shift, carry into the exponent, overflow to infinity. sig == 0 returns a zero of the given sign.
// Callers that jam a sticky bit into sig[0] must keep it at least two bits below the rounding position
// whenever it is set (a50_add32_align guarantees this; a50_bf16_mul never jams).
module a50_round_pack (
    input  wire               sign,
    input  wire [27:0]        sig,
    input  wire signed [11:0] lsb_exp,
    output reg  [31:0]        word
);
    integer i, top, le, top_exp, quantum, sh;
    reg [27:0] mask, rem, half;
    /* verilator lint_off UNUSEDSIGNAL */
    reg [27:0] t;                                // only t[24:0] is used
    /* verilator lint_on UNUSEDSIGNAL */
    reg [24:0] r;
    reg up;
    always @* begin
        top = 0;
        for (i = 0; i < 28; i = i + 1)
            if (sig[i]) top = i;
        le = {{20{lsb_exp[11]}}, lsb_exp};       // sign-extend
        top_exp = top + le;
        quantum = top_exp - 23;
        if (quantum < -149) quantum = -149;
        sh = quantum - le;
        mask = 28'd0; rem = 28'd0; half = 28'd0; t = 28'd0; up = 1'b0; r = 25'd0;
        if (sh <= 0) begin
            t = sig << (-sh);
            r = t[24:0];                         // exact; the leading one lands at or below bit 23
        end else if (sh <= 28) begin
            mask = (28'd1 << sh) - 28'd1;
            rem  = sig & mask;
            half = 28'd1 << (sh - 1);
            t    = sig >> sh;
            r    = t[24:0];
            up   = (rem > half) || (rem == half && r[0]);
            r    = r + {24'd0, up};
        end else begin
            r = 25'd0;                           // sig < 2^28 <= half: rounds to zero
        end
        if (sig == 28'd0 || r == 25'd0)
            word = {sign, 31'd0};
        else if (r < 25'h0800000)
            word = {sign, 8'd0, r[22:0]};        // subnormal (quantum == -149)
        else begin
            if (r == 25'h1000000) begin          // rounding carried out of 24 bits
                r = 25'h0800000;
                quantum = quantum + 1;
            end
            if (quantum + 150 >= 255)
                word = {sign, 8'hFF, 23'd0};
            else
                word = {sign, quantum[7:0] + 8'd150, r[22:0]};
        end
    end
endmodule
