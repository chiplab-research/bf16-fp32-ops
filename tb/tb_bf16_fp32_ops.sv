// Self-checking bench for the bf16_fp32_ops operators.
// Reads vectors written by run_tests.py and compares every output bit-exact:
//   mul.hex: "aaaa bbbb pppppppp"          a50_bf16_mul:  p = RN32(bf16 a * bf16 b)
//   add.hex: "xxxxxxxx yyyyyyyy ssssssss"  a50_add32_align -> a50_round_pack: s = RN32(x + y)
// The two adder stages are connected combinationally here (in a pipeline a register goes between them;
// that does not change the value). Prints one RESULT line; the runner parses it.
module tb_bf16_fp32_ops;
    reg  [15:0] a, b;
    reg  [31:0] x, y;
    reg  [31:0] p_exp, s_exp;
    wire [31:0] p, s, rounded, special_word;
    wire special, sign;
    wire [27:0] sum;
    wire signed [11:0] lsb_exp;

    a50_bf16_mul    u_mul(.a(a), .b(b), .p(p));
    a50_add32_align u_align(.a(x), .b(y), .special(special), .special_word(special_word),
                            .sign(sign), .sum(sum), .lsb_exp(lsb_exp));
    a50_round_pack  u_round(.sign(sign), .sig(sum), .lsb_exp(lsb_exp), .word(rounded));
    assign s = special ? special_word : rounded;

    integer fd, n, mul_n, mul_bad, add_n, add_bad;
    initial begin
        mul_n = 0; mul_bad = 0; add_n = 0; add_bad = 0;
        a = 0; b = 0; x = 0; y = 0;
        fd = $fopen("mul.hex", "r");
        if (fd == 0) begin $display("ERROR cannot open mul.hex"); $finish; end
        n = $fscanf(fd, "%h %h %h\n", a, b, p_exp);
        while (n == 3) begin
            #1;
            mul_n = mul_n + 1;
            if (p !== p_exp) begin
                mul_bad = mul_bad + 1;
                if (mul_bad <= 10) $display("MUL MISMATCH %04h * %04h got %08h expected %08h", a, b, p, p_exp);
            end
            n = $fscanf(fd, "%h %h %h\n", a, b, p_exp);
        end
        $fclose(fd);
        fd = $fopen("add.hex", "r");
        if (fd == 0) begin $display("ERROR cannot open add.hex"); $finish; end
        n = $fscanf(fd, "%h %h %h\n", x, y, s_exp);
        while (n == 3) begin
            #1;
            add_n = add_n + 1;
            if (s !== s_exp) begin
                add_bad = add_bad + 1;
                if (add_bad <= 10) $display("ADD MISMATCH %08h + %08h got %08h expected %08h", x, y, s, s_exp);
            end
            n = $fscanf(fd, "%h %h %h\n", x, y, s_exp);
        end
        $fclose(fd);
        $display("RESULT mul=%0d mul_bad=%0d add=%0d add_bad=%0d %s", mul_n, mul_bad, add_n, add_bad,
                 (mul_bad == 0 && add_bad == 0 && mul_n > 0 && add_n > 0) ? "PASS" : "FAIL");
        $finish;
    end
endmodule
