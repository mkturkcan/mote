	.arch armv8.2-a+crc+fp16+rcpc+dotprod
	.file	"q3gemv.c"
	.text
	.align	2
	.p2align 5,,15
	.global	q3_8x16_chunk
	.type	q3_8x16_chunk, %function
q3_8x16_chunk:
.LFB4340:
	.cfi_startproc
	ldr	d7, [x1]
	movi	v5.16b, 0x1
	movi	v4.16b, 0x3
	movi	v2.16b, 0x4
	ldr	d6, [x0]
	ldp	q23, q22, [x2]
	dup	d3, v7.d[0]
	ldp	q20, q19, [x2, 32]
	dup	d1, v6.d[0]
	ldp	q17, q18, [x2, 64]
	ins	v3.d[1], v7.d[0]
	ins	v1.d[1], v6.d[0]
	ldp	q6, q7, [x2, 96]
	ushr	v21.16b, v3.16b, 1
	and	v3.16b, v3.16b, v5.16b
	ushr	v16.16b, v1.16b, 2
	and	v1.16b, v1.16b, v4.16b
	and	v21.16b, v21.16b, v5.16b
	shl	v3.16b, v3.16b, 2
	and	v16.16b, v16.16b, v4.16b
	shl	v21.16b, v21.16b, 2
	orr	v1.16b, v1.16b, v3.16b
	orr	v3.16b, v16.16b, v21.16b
	sub	v1.16b, v1.16b, v2.16b
	mov	v16.16b, v23.16b
	sub	v3.16b, v3.16b, v2.16b
	sdot	v16.4s, v1.16b, v0.16b
	mov	v1.16b, v22.16b
	sdot	v1.4s, v3.16b, v0.16b
	stp	q16, q1, [x2]
	ldr	d21, [x1, 8]
	ldr	d16, [x0, 16]
	dup	d3, v21.d[0]
	dup	d1, v16.d[0]
	ins	v3.d[1], v21.d[0]
	ins	v1.d[1], v16.d[0]
	ushr	v21.16b, v3.16b, 1
	and	v3.16b, v3.16b, v5.16b
	ushr	v16.16b, v1.16b, 2
	and	v1.16b, v1.16b, v4.16b
	and	v21.16b, v21.16b, v5.16b
	shl	v3.16b, v3.16b, 2
	and	v16.16b, v16.16b, v4.16b
	shl	v21.16b, v21.16b, 2
	orr	v1.16b, v1.16b, v3.16b
	orr	v3.16b, v16.16b, v21.16b
	sub	v1.16b, v1.16b, v2.16b
	sub	v3.16b, v3.16b, v2.16b
	sdot	v20.4s, v1.16b, v0.16b
	sdot	v19.4s, v3.16b, v0.16b
	stp	q20, q19, [x2, 32]
	ldr	d19, [x1, 16]
	ldr	d16, [x0, 32]
	dup	d3, v19.d[0]
	dup	d1, v16.d[0]
	ins	v3.d[1], v19.d[0]
	ins	v1.d[1], v16.d[0]
	ushr	v19.16b, v3.16b, 1
	and	v3.16b, v3.16b, v5.16b
	ushr	v16.16b, v1.16b, 2
	and	v1.16b, v1.16b, v4.16b
	and	v19.16b, v19.16b, v5.16b
	shl	v3.16b, v3.16b, 2
	and	v16.16b, v16.16b, v4.16b
	shl	v19.16b, v19.16b, 2
	orr	v1.16b, v1.16b, v3.16b
	orr	v3.16b, v16.16b, v19.16b
	sub	v1.16b, v1.16b, v2.16b
	sub	v3.16b, v3.16b, v2.16b
	sdot	v17.4s, v1.16b, v0.16b
	sdot	v18.4s, v3.16b, v0.16b
	stp	q17, q18, [x2, 64]
	ldr	d17, [x1, 24]
	ldr	d16, [x0, 48]
	dup	d3, v17.d[0]
	dup	d1, v16.d[0]
	ins	v3.d[1], v17.d[0]
	ins	v1.d[1], v16.d[0]
	ushr	v17.16b, v3.16b, 1
	and	v3.16b, v3.16b, v5.16b
	ushr	v16.16b, v1.16b, 2
	and	v1.16b, v1.16b, v4.16b
	and	v5.16b, v17.16b, v5.16b
	shl	v3.16b, v3.16b, 2
	and	v4.16b, v16.16b, v4.16b
	shl	v5.16b, v5.16b, 2
	orr	v1.16b, v1.16b, v3.16b
	orr	v3.16b, v4.16b, v5.16b
	sub	v1.16b, v1.16b, v2.16b
	sub	v2.16b, v3.16b, v2.16b
	sdot	v6.4s, v1.16b, v0.16b
	sdot	v7.4s, v2.16b, v0.16b
	stp	q6, q7, [x2, 96]
	ret
	.cfi_endproc
.LFE4340:
	.size	q3_8x16_chunk, .-q3_8x16_chunk
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
