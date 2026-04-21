	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii ; -- Begin function _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii
	.globl	_Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii
	.p2align	8
	.type	_Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii,@function
_Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii: ; @_Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii
; %bb.0:                                ; %.preheader86
	s_load_b64 s[4:5], s[0:1], 0x18
	v_lshrrev_b32_e32 v1, 5, v0
	s_lshl_b32 s2, s2, 3
	s_mov_b32 s6, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_lshl_add_u32 v8, v1, 1, s2
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s5, s5, 7
	s_cmp_gt_i32 s5, 0
	s_cbranch_scc1 .LBB0_2
; %bb.1:                                ; %.preheader86..preheader_crit_edge
	v_lshl_add_u32 v1, v1, 1, s2
	s_branch .LBB0_3
.LBB0_2:
	s_mov_b32 s6, -1
                                        ; implicit-def: $vgpr1
.LBB0_3:                                ; %Flow149
	s_load_b64 s[2:3], s[0:1], 0x10
	v_dual_mov_b32 v10, 0 :: v_dual_and_b32 v7, 31, v0
	v_mov_b32_e32 v9, 0
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_7
; %bb.4:                                ; %.lr.ph
	s_load_b128 s[8:11], s[0:1], 0x0
	v_or_b32_e32 v1, 1, v8
	v_cmp_gt_i32_e64 s0, s4, v8
	s_mul_i32 s1, s5, 34
	v_dual_mov_b32 v10, 0 :: v_dual_lshlrev_b32 v11, 3, v7
	v_cmp_gt_i32_e32 vcc_lo, s4, v1
	v_cndmask_b32_e64 v6, 0, v8, s0
	v_dual_mov_b32 v9, 0 :: v_dual_lshlrev_b32 v0, 1, v0
	v_cndmask_b32_e32 v5, 0, v1, vcc_lo
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_mad_u64_u32 v[1:2], null, v5, s1, s[8:9]
	v_mad_u64_u32 v[3:4], null, v6, s1, s[8:9]
	v_ashrrev_i32_e32 v5, 31, v5
	v_ashrrev_i32_e32 v6, 31, v6
	v_mad_u64_u32 v[14:15], null, v5, s1, v[2:3]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_mad_u64_u32 v[4:5], null, v6, s1, v[4:5]
	v_add_co_u32 v12, s1, v7, 2
	v_add_co_ci_u32_e64 v13, null, 0, 0, s1
	v_add_co_u32 v5, s1, s10, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v6, null, s11, 0, s1
	v_mov_b32_e32 v2, v14
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_barrier
	buffer_gl0_inv
	global_load_d16_b16 v14, v[5:6], off
	v_add_co_u32 v15, s1, v3, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v16, null, v4, v13, s1
	v_add_co_u32 v17, s1, v1, v12
	v_add_co_ci_u32_e64 v18, null, v2, v13, s1
	v_add_co_u32 v5, s1, 0x100, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v6, null, 0, v6, s1
	s_add_i32 s5, s5, -1
	s_cmp_eq_u32 s5, 0
	s_waitcnt vmcnt(0)
	ds_store_b16 v0, v14
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	global_load_u8 v16, v[15:16], off
	global_load_u8 v17, v[17:18], off
	global_load_d16_b16 v18, v[3:4], off
	global_load_d16_b16 v19, v[1:2], off
	ds_load_b64 v[14:15], v11
	s_waitcnt lgkmcnt(0)
	v_cvt_f32_f16_e32 v20, v14.l
	s_waitcnt vmcnt(3)
	v_cmp_gt_u32_e64 s1, 64, v16
	v_and_b32_e32 v21, 3, v16
	v_bfe_u32 v22, v16, 2, 2
	v_bfe_u32 v23, v16, 4, 2
	v_and_b32_e32 v24, 0xc0, v16
	v_cndmask_b32_e64 v16, 0, 1.0, s1
	s_waitcnt vmcnt(2)
	v_cmp_gt_u32_e64 s1, 64, v17
	v_and_b32_e32 v25, 3, v17
	v_bfe_u32 v26, v17, 2, 2
	v_bfe_u32 v27, v17, 4, 2
	v_and_b32_e32 v28, 0xc0, v17
	v_cndmask_b32_e64 v17, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 2, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v29, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 0, v21
	v_cndmask_b32_e64 v21, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 2, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v21, v29, v21
	v_cndmask_b32_e64 v30, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 0, v22
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v21, v21, v20
	v_cndmask_b32_e64 v22, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 2, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v22, v30, v22
	v_cndmask_b32_e64 v31, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 0, v23
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_mix_f32 v21, v22, v14, v21 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_cndmask_b32_e64 v23, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 0x80, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v23, v31, v23
	v_cndmask_b32_e64 v24, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 2, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_sub_f32_e32 v16, v24, v16
	v_cndmask_b32_e64 v32, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 0, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v25, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 2, v26
	v_sub_f32_e32 v25, v32, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v33, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 0, v26
	v_mul_f32_e32 v20, v25, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v26, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 2, v27
	v_sub_f32_e32 v26, v33, v26
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v34, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 0, v27
	v_fma_mix_f32 v14, v26, v14, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v20, v23, v15, v21 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e64 v27, 0, 1.0, s1
	v_cmp_eq_u32_e64 s1, 0x80, v28
	v_fma_mix_f32 v16, v16, v15, v20 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sub_f32_e32 v25, v34, v27
	v_cndmask_b32_e64 v28, 0, 1.0, s1
	v_add_co_u32 v1, s1, v1, 34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v2, null, 0, v2, s1
	v_sub_f32_e32 v17, v28, v17
	v_fma_mix_f32 v14, v25, v15, v14 op_sel_hi:[0,1,0]
	v_add_co_u32 v3, s1, v3, 34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v4, null, 0, v4, s1
	v_fma_mix_f32 v14, v17, v15, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(1)
	v_fma_mix_f32 v15, v18, v16, v10 op_sel_hi:[1,0,0]
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_mix_f32 v14, v19, v14, v9 op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v10, v10, v15, s0
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v9, v9, v14, vcc_lo
	s_cbranch_scc0 .LBB0_5
; %bb.6:                                ; %Flow148
	v_mov_b32_e32 v1, v8
.LBB0_7:                                ; %Flow150
	v_mbcnt_lo_u32_b32 v2, -1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e64 s0, s4, v1
	v_xor_b32_e32 v0, 16, v2
	v_xor_b32_e32 v3, 8, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cmp_gt_i32_e32 vcc_lo, 32, v0
	v_cndmask_b32_e32 v0, v2, v0, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	v_cndmask_b32_e32 v3, v2, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b32_e32 v3, 2, v3
	v_lshlrev_b32_e32 v0, 2, v0
	ds_bpermute_b32 v4, v0, v10
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v5, v10, v4
	v_xor_b32_e32 v4, 4, v2
	v_xor_b32_e32 v10, 1, v2
	ds_bpermute_b32 v6, v3, v5
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	v_cndmask_b32_e32 v4, v2, v4, vcc_lo
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v6, v5, v6
	v_xor_b32_e32 v5, 2, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	v_cndmask_b32_e32 v5, v2, v5, vcc_lo
	v_cmp_gt_i32_e32 vcc_lo, 32, v10
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshlrev_b32_e32 v4, 2, v4
	v_cndmask_b32_e32 v10, v2, v10, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v7
	ds_bpermute_b32 v8, v4, v6
	s_and_b32 s0, vcc_lo, s0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v6, v6, v8
	ds_bpermute_b32 v8, v5, v6
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v6, v8
	v_lshlrev_b32_e32 v6, 2, v10
	ds_bpermute_b32 v8, v6, v2
	s_and_saveexec_b32 s1, s0
	s_cbranch_execz .LBB0_9
; %bb.8:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v2, v2, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_nlt_f32_e64 s0, 0x477fe000, v2
	v_cndmask_b32_e64 v7, 0x477fe000, v2, s0
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ngt_f32_e64 s0, 0xc77fe000, v7
	v_cndmask_b32_e64 v10, 0xc77fe000, v7, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64 v[7:8], 1, v[1:2]
	v_cvt_f16_f32_e32 v2.l, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v7, s0, s2, v7
	v_add_co_ci_u32_e64 v8, null, s3, v8, s0
	global_store_b16 v[7:8], v2, off
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s1
	ds_bpermute_b32 v0, v0, v9
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v9, v0
	ds_bpermute_b32 v2, v3, v0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v2
	ds_bpermute_b32 v2, v4, v0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v2
	ds_bpermute_b32 v2, v5, v0
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v2
	ds_bpermute_b32 v2, v6, v0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_12
; %bb.10:
	v_add_nc_u32_e32 v3, 1, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s4, v3
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_12
; %bb.11:
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v2
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cmp_nlt_f32_e32 vcc_lo, 0x477fe000, v0
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	v_cndmask_b32_e32 v0, 0x477fe000, v0, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cmp_ngt_f32_e32 vcc_lo, 0xc77fe000, v0
	v_cndmask_b32_e32 v0, 0xc77fe000, v0, vcc_lo
	v_add_co_u32 v1, vcc_lo, s2, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v2, null, s3, v2, vcc_lo
	v_cvt_f16_f32_e32 v0.l, v0
	global_store_b16 v[1:2], v0, off offset:2
.LBB0_12:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 35
		.amdhsa_next_free_sgpr 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size 12
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii, .Lfunc_end0-_Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii
                                        ; -- End function
	.set _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.num_vgpr, 35
	.set _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.num_agpr, 0
	.set _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.numbered_sgpr, 12
	.set _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.num_named_barrier, 0
	.set _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.private_seg_size, 0
	.set _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.uses_vcc, 1
	.set _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.uses_flat_scratch, 0
	.set _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.has_dyn_sized_stack, 0
	.set _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.has_recursion, 0
	.set _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1476
; TotalNumSgprs: 14
; NumVgprs: 35
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 256 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 35
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_aa3faf3571d05223,@object ; @__hip_cuid_aa3faf3571d05223
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_aa3faf3571d05223
__hip_cuid_aa3faf3571d05223:
	.byte	0                               ; 0x0
	.size	__hip_cuid_aa3faf3571d05223, 1

	.ident	"AMD clang version 22.0.0git (/startdir/rocm-llvm f58b06dce1f9c15707c5f808fd002e18c2accf7e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_aa3faf3571d05223
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 256
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 128
    .name:           _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z22bonsai_tq2_gemv_kernelPKhPK6__halfPS1_ii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     35
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
