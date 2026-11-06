module Tcc

using FuzzifiED
using FuzzifiED.SO3lver
using LinearAlgebra
using SparseArrays
using Serialization
using SHA
using Random
using TOML
import KrylovKit

FuzzifiED.ElementType = Float64

include("version.jl")

include("so5_c2_density_phase.jl")
using .SO5C2DensityPhase: cross_density_coefficient
include("digest.jl")
include("gates.jl")
include("model.jl")
include("resources.jl")
include("ph.jl")
include("seglib.jl")
include("exchange.jl")
include("telemetry.jl")
include("matvec.jl")
include("runner.jl")
include("fast_simplify.jl")
include("ZhouGenerator.jl")
include("edges.jl")
include("targets.jl")
include("kkstage.jl")
include("tuning.jl")
include("ope.jl")
include("ope3.jl")
include("parity.jl")
include("pipeline.jl")
include("stageb.jl")
include("fullq.jl")
include("post.jl")
include("plan.jl")

export segment_setup, build_segment_spaces, ph_build_half, ph_expand, ph_half_default, build_terms, hamiltonian_template, set_g!, load_segment_spaces
export SegLibrary, segops!, segop!, segops_shared!, segop_shared!, absorb!, scaled_segop, release!, resident_bytes, tiled_seg_operator, segop_difference
export ExchangeSector, exchange_map, sector_reduce!, sector_expand!, reduced_action
export enable_fast_simplify!
export matvec_plan, tcc_mul!, tcc_mul_block!, MatvecPlan, act_block!, NATIVE_DIM_MAX
export edge_arrays, edge_arrays_incremental, column_fingerprints, edge_arrays_reference, gamma2_terms, kk_terms
export TCC_VERSION, file_digest, tree_digest, git_commit
export build_half_segment, so5_types
export run_tune_stage, find_root, tune_state, TUNE_TARGETS
export run_pipeline, pipeline_config, tuning_F, run_rows, run_kk_stage, run_ope_stage, source_window, target_window
export extend_targets, target_requests, window_frontier, resolve_c2!, polish_c2!
export run_spectrum, lowest_deflated, pair_schedule, split_segment_spaces, post_h, post_c
export segment_sector_dim_max, resolve_residency, residency_auto, PH_HALF_DIM_MIN, RESIDENCY_PAIR_BYTES
export interleave_memory!
export memory_kib, cpu_seconds, Stage, stage!
export run_ope3, ope3_spaces, qg_terms_gate, spin_factor, tensor_g_fierz_couplings, ope3_terms
export segment_ph_maps, comp_parity, apply_parity, parity_values, physical_parity
export GATE, GateError, window_gate, column_shas, producer_info
export run_steps, expand_plan!, stageb_options, run_stageb, run_rawtarget, run_fullq, run_coverage, run_targets_pairs, run_ope3_step, run_parity
export tensor_g_parts, insertion_terms

end
