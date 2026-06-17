	.arch armv8.2-a+crc+fp16+rcpc+dotprod
	.file	"q3gemv2.c"
	.text
	.align	2
	.p2align 5,,15
	.global	q3_gemv_8
	.type	q3_gemv_8, %function
q3_gemv_8:
.LFB4340:
	.cfi_startproc
	stp	d8, d9, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 72, -32
	.cfi_offset 73, -24
	cmp	w3, 0
	ble	.L4
	movi	v21.4s, 0
	movi	v17.16b, 0x1
	mov	w8, 0
	movi	v16.16b, 0x3
	movi	v7.16b, 0x4
	stp	d10, d11, [sp, 16]
	.cfi_offset 75, -8
	.cfi_offset 74, -16
	mov	v23.16b, v21.16b
	mov	v25.16b, v21.16b
	mov	v19.16b, v21.16b
	mov	v22.16b, v21.16b
	mov	v24.16b, v21.16b
	mov	v26.16b, v21.16b
	mov	v20.16b, v21.16b
	.p2align 5,,15
.L3:
	asr	w6, w8, 4
	ldr	q0, [x2], 16
	add	w8, w8, 16
	lsl	w9, w6, 5
	lsl	w6, w6, 6
	sxtw	x7, w9
	sxtw	x5, w6
	add	x12, x7, 8
	ldr	q3, [x1, w9, sxtw]
	add	x11, x7, 16
	add	x7, x7, 24
	add	x10, x5, 16
	ldr	q2, [x1, x12]
	add	x9, x5, 32
	add	x5, x5, 48
	ldr	q1, [x1, x11]
	ldr	q29, [x1, x7]
	ushr	v10.16b, v3.16b, 1
	and	v3.16b, v3.16b, v17.16b
	ushr	v18.16b, v2.16b, 1
	and	v2.16b, v2.16b, v17.16b
	ldr	q11, [x0, x10]
	ldr	q6, [x0, w6, sxtw]
	ushr	v9.16b, v1.16b, 1
	and	v1.16b, v1.16b, v17.16b
	shl	v8.16b, v3.16b, 2
	and	v10.16b, v10.16b, v17.16b
	shl	v27.16b, v2.16b, 2
	ldr	q4, [x0, x9]
	and	v3.16b, v18.16b, v17.16b
	ushr	v31.16b, v29.16b, 1
	and	v29.16b, v29.16b, v17.16b
	ldr	q2, [x0, x5]
	ushr	v28.16b, v11.16b, 2
	and	v9.16b, v9.16b, v17.16b
	shl	v30.16b, v1.16b, 2
	and	v11.16b, v11.16b, v16.16b
	shl	v18.16b, v3.16b, 2
	ushr	v5.16b, v6.16b, 2
	and	v31.16b, v31.16b, v17.16b
	ushr	v3.16b, v4.16b, 2
	and	v6.16b, v6.16b, v16.16b
	shl	v10.16b, v10.16b, 2
	and	v28.16b, v28.16b, v16.16b
	ushr	v1.16b, v2.16b, 2
	and	v4.16b, v4.16b, v16.16b
	and	v5.16b, v5.16b, v16.16b
	shl	v9.16b, v9.16b, 2
	and	v2.16b, v2.16b, v16.16b
	and	v3.16b, v3.16b, v16.16b
	shl	v29.16b, v29.16b, 2
	orr	v28.16b, v28.16b, v18.16b
	shl	v31.16b, v31.16b, 2
	and	v1.16b, v1.16b, v16.16b
	orr	v18.16b, v11.16b, v27.16b
	orr	v6.16b, v6.16b, v8.16b
	orr	v5.16b, v5.16b, v10.16b
	orr	v4.16b, v4.16b, v30.16b
	orr	v3.16b, v3.16b, v9.16b
	orr	v2.16b, v2.16b, v29.16b
	orr	v1.16b, v1.16b, v31.16b
	sub	v27.16b, v28.16b, v7.16b
	sub	v18.16b, v18.16b, v7.16b
	sub	v6.16b, v6.16b, v7.16b
	sub	v5.16b, v5.16b, v7.16b
	sub	v4.16b, v4.16b, v7.16b
	sub	v3.16b, v3.16b, v7.16b
	sub	v2.16b, v2.16b, v7.16b
	sub	v1.16b, v1.16b, v7.16b
	sdot	v22.4s, v27.16b, v0.16b
	sdot	v24.4s, v18.16b, v0.16b
	sdot	v20.4s, v6.16b, v0.16b
	sdot	v26.4s, v5.16b, v0.16b
	sdot	v19.4s, v4.16b, v0.16b
	sdot	v25.4s, v3.16b, v0.16b
	sdot	v23.4s, v2.16b, v0.16b
	sdot	v21.4s, v1.16b, v0.16b
	cmp	w3, w8
	bgt	.L3
	ldp	d10, d11, [sp, 16]
	.cfi_restore 75
	.cfi_restore 74
.L2:
	addv	s26, v26.4s
	addv	s25, v25.4s
	addv	s20, v20.4s
	addv	s19, v19.4s
	ldp	d8, d9, [sp], 32
	.cfi_remember_state
	.cfi_restore 73
	.cfi_restore 72
	.cfi_def_cfa_offset 0
	addv	s24, v24.4s
	addv	s23, v23.4s
	addv	s22, v22.4s
	addv	s21, v21.4s
	ins	v20.s[1], v26.s[0]
	ins	v19.s[1], v25.s[0]
	ins	v20.s[2], v24.s[0]
	ins	v19.s[2], v23.s[0]
	ins	v20.s[3], v22.s[0]
	ins	v19.s[3], v21.s[0]
	stp	q20, q19, [x4]
	ret
	.p2align 2,,3
.L4:
	.cfi_restore_state
	movi	v21.4s, 0
	mov	v23.16b, v21.16b
	mov	v25.16b, v21.16b
	mov	v19.16b, v21.16b
	mov	v22.16b, v21.16b
	mov	v24.16b, v21.16b
	mov	v26.16b, v21.16b
	mov	v20.16b, v21.16b
	b	.L2
	.cfi_endproc
.LFE4340:
	.size	q3_gemv_8, .-q3_gemv_8
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
