// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Portable MoE / QMoE Offload v4 Session Policy, Source Descriptors, Capability Reports, and Execution Plan.

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/common/status.h"
#include "core/framework/config_options.h"
#include "core/framework/tensor_shape.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime::moe_offload {

enum class OffloadMode : uint8_t {
  kDisabled = 0,
  kAuto = 1,
  kRequired = 2
};

enum class CpuExecutionMode : uint8_t {
  kAuto = 0,
  kEnabled = 1,
  kDisabled = 2
};

enum class PlacementPolicy : uint8_t {
  kStatic = 0,
  kAdaptive = 1
};

struct BudgetSpec {
  bool is_auto{true};
  uint64_t bytes{0};  // Valid when !is_auto; 0 means zero capacity (never "unlimited")
};

struct NodeOverrideSpec {
  std::string graph_scope{"main"};
  std::string node_name;
  std::optional<uint64_t> device_expert_bytes;
  std::optional<uint64_t> host_expert_bytes;
  std::optional<int32_t> max_resident_slots;
};

struct MoeExternalInitializerInfo {
  std::filesystem::path file_path;
  uint64_t file_offset{0};
  uint64_t byte_length{0};
};

struct MoeOffloadSessionPolicy {
  OffloadMode mode{OffloadMode::kDisabled};
  CpuExecutionMode cpu_execution{CpuExecutionMode::kAuto};
  PlacementPolicy placement{PlacementPolicy::kStatic};

  BudgetSpec device_expert_bytes{true, 0};
  BudgetSpec host_expert_bytes{true, 0};
  uint64_t device_working_bytes{256ULL << 20};  // Default 256 MiB transient GPU workspace cap
  uint64_t host_working_bytes{128ULL << 20};    // Default 128 MiB transient host workspace cap
  uint64_t pinned_bytes{256ULL << 20};          // Default 256 MiB unique pinned/registered host cap
  uint64_t reserve_device_bytes{0};             // External KV/state/capture/child-session reserve
  std::string budget_group{"default"};          // Maps to CudaDeviceMemoryBudget::BudgetGroup

  bool allow_file_backing{true};
  int32_t cpu_threads{0};            // 0 = auto physical-core discovery
  int32_t async_max_tokens{4};       // Decode vs. prefill streaming token threshold
  int32_t prefill_wave_experts{32};  // Max active cold experts per DMA wave

  std::filesystem::path source_root;
  std::filesystem::path derived_cache_dir;
  std::filesystem::path popularity_file;
  std::vector<NodeOverrideSpec> node_overrides;

  static std::string ToLowerAscii(std::string_view s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
  }

  static bool TryParseUint64(std::string_view s, uint64_t& value) noexcept {
    if (s.empty()) return false;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), value, 10);
    return res.ec == std::errc{} && res.ptr == s.data() + s.size();
  }

  static bool TryParseInt32(std::string_view s, int32_t& value) noexcept {
    if (s.empty()) return false;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), value, 10);
    return res.ec == std::errc{} && res.ptr == s.data() + s.size();
  }

  static bool TryParseBudgetSpec(std::string_view s, BudgetSpec& spec) noexcept {
    const std::string lower = ToLowerAscii(s);
    if (lower == "auto") {
      spec.is_auto = true;
      spec.bytes = 0;
      return true;
    }
    uint64_t parsed = 0;
    if (!TryParseUint64(s, parsed)) return false;
    spec.is_auto = false;
    spec.bytes = parsed;
    return true;
  }

  static std::filesystem::path ResolveRelativePath(const std::filesystem::path& base_dir,
                                                   const std::string& raw_path) {
    if (raw_path.empty()) return {};
    std::filesystem::path p(raw_path);
    if (p.is_relative() && !base_dir.empty()) {
      std::error_code ec;
      auto canon = std::filesystem::weakly_canonical(base_dir / p, ec);
      if (!ec && !canon.empty()) return canon;
      return base_dir / p;
    }
    return p;
  }

  static common::Status ParseFromConfigOptions(const ConfigOptions& config_options,
                                               const std::filesystem::path& model_location,
                                               MoeOffloadSessionPolicy& out) noexcept {
    out = MoeOffloadSessionPolicy{};
    std::filesystem::path base_dir;
    if (!model_location.empty()) {
      std::error_code ec;
      if (std::filesystem::is_directory(model_location, ec)) {
        base_dir = model_location;
      } else {
        base_dir = model_location.parent_path();
      }
    }

    const auto& kv_map = config_options.GetConfigOptionsMap();
    for (const auto& [key, val] : kv_map) {
      if (key.rfind("moe.cpu_offload.", 0) == 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Prototype configuration key '", key,
                               "' is discarded and not supported in QMoE Offload v4; use 'session.moe.offload.*'.");
      }
      if (key.rfind("session.moe.offload.", 0) != 0) {
        continue;
      }
      if (key == kOrtSessionOptionsMoeOffloadMode) {
        const std::string v = ToLowerAscii(val);
        if (v == "disabled" || v == "0" || v == "off") {
          out.mode = OffloadMode::kDisabled;
        } else if (v == "auto") {
          out.mode = OffloadMode::kAuto;
        } else if (v == "required" || v == "1" || v == "on") {
          out.mode = OffloadMode::kRequired;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadMode, " value '", val,
                                 "'; expected 'disabled', 'auto', or 'required'.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadCpuExecution) {
        const std::string v = ToLowerAscii(val);
        if (v == "auto") {
          out.cpu_execution = CpuExecutionMode::kAuto;
        } else if (v == "enabled" || v == "1") {
          out.cpu_execution = CpuExecutionMode::kEnabled;
        } else if (v == "disabled" || v == "0") {
          out.cpu_execution = CpuExecutionMode::kDisabled;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadCpuExecution, " value '", val,
                                 "'; expected 'auto', 'enabled', or 'disabled'.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadPlacement) {
        const std::string v = ToLowerAscii(val);
        if (v == "static") {
          out.placement = PlacementPolicy::kStatic;
        } else if (v == "adaptive") {
          out.placement = PlacementPolicy::kAdaptive;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadPlacement, " value '", val,
                                 "'; expected 'static' or 'adaptive' (prototype 'basic'/'layered' are not supported).");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadDeviceExpertBytes) {
        if (!TryParseBudgetSpec(val, out.device_expert_bytes)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadDeviceExpertBytes, " value '", val,
                                 "'; expected 'auto' or non-negative integer bytes.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadHostExpertBytes) {
        if (!TryParseBudgetSpec(val, out.host_expert_bytes)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadHostExpertBytes, " value '", val,
                                 "'; expected 'auto' or non-negative integer bytes.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadDeviceWorkingBytes) {
        if (!TryParseUint64(val, out.device_working_bytes) || out.device_working_bytes == 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadDeviceWorkingBytes, " value '", val,
                                 "'; expected positive integer bytes.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadHostWorkingBytes) {
        if (!TryParseUint64(val, out.host_working_bytes) || out.host_working_bytes == 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadHostWorkingBytes, " value '", val,
                                 "'; expected positive integer bytes.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadPinnedBytes) {
        if (!TryParseUint64(val, out.pinned_bytes)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadPinnedBytes, " value '", val,
                                 "'; expected non-negative integer bytes.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadReserveDeviceBytes) {
        if (ToLowerAscii(val) == "auto") {
          out.reserve_device_bytes = 0;
        } else if (!TryParseUint64(val, out.reserve_device_bytes)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadReserveDeviceBytes, " value '", val,
                                 "'; expected 'auto' or non-negative integer bytes.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadBudgetGroup) {
        if (val.empty()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 kOrtSessionOptionsMoeOffloadBudgetGroup, " must not be empty.");
        }
        out.budget_group = val;
      } else if (key == kOrtSessionOptionsMoeOffloadAllowFileBacking) {
        if (val == "1" || ToLowerAscii(val) == "true") {
          out.allow_file_backing = true;
        } else if (val == "0" || ToLowerAscii(val) == "false") {
          out.allow_file_backing = false;
        } else {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadAllowFileBacking, " value '", val,
                                 "'; expected '0' or '1'.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadCpuThreads) {
        if (!TryParseInt32(val, out.cpu_threads) || out.cpu_threads < 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadCpuThreads, " value '", val,
                                 "'; expected non-negative integer.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadAsyncMaxTokens) {
        if (!TryParseInt32(val, out.async_max_tokens) || out.async_max_tokens < 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadAsyncMaxTokens, " value '", val,
                                 "'; expected non-negative integer.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadPrefillWaveExperts) {
        if (!TryParseInt32(val, out.prefill_wave_experts) || out.prefill_wave_experts <= 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Invalid ", kOrtSessionOptionsMoeOffloadPrefillWaveExperts, " value '", val,
                                 "'; expected positive integer.");
        }
      } else if (key == kOrtSessionOptionsMoeOffloadPopularityFile) {
        out.popularity_file = ResolveRelativePath(base_dir, val);
      } else if (key == kOrtSessionOptionsMoeOffloadSourceRoot) {
        out.source_root = ResolveRelativePath(base_dir, val);
      } else if (key == kOrtSessionOptionsMoeOffloadDerivedCacheDir) {
        out.derived_cache_dir = ResolveRelativePath(base_dir, val);
      } else if (key == kOrtSessionOptionsMoeOffloadNodeOverrides) {
        // Optional node overrides JSON string; empty is allowed.
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Unknown session.moe.offload.* configuration key: ", key);
      }
    }

    if (out.source_root.empty() && !base_dir.empty()) {
      out.source_root = base_dir;
    }
    return common::Status::OK();
  }
};

enum class ExpertBankKind : uint8_t {
  kFc1Weights = 0,
  kFc2Weights = 1,
  kFc1Scales = 2,
  kFc2Scales = 3,
  kFc1GlobalScale = 4,
  kFc2GlobalScale = 5,
  kFc1Bias = 6,
  kFc2Bias = 7,
  kFc1ZeroPoints = 8,
  kFc2ZeroPoints = 9,
  kCount = 10
};

struct MoeExpertBankSlice {
  ExpertBankKind kind{};
  int input_index{-1};
  std::string initializer_name;
  ONNXTensorElementDataType elem_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  TensorShape full_shape;
  size_t bytes_per_expert{0};
  const void* host_data{nullptr};
  size_t total_bytes{0};
  std::filesystem::path external_file_path;
  uint64_t external_file_offset{0};
};

struct MoeExpertSourceDescriptor {
  std::string graph_scope{"main"};
  std::string stable_node_id;
  std::string op_type;
  std::string quant_type;
  int32_t layer_index{-1};
  int32_t num_experts{0};
  int32_t top_k{0};
  int32_t hidden_size{0};
  int32_t inter_size{0};
  int32_t block_size{0};
  int32_t weight_bits{0};
  bool weights_prepacked{false};
  std::array<std::optional<MoeExpertBankSlice>, static_cast<size_t>(ExpertBankKind::kCount)> banks;
};

struct MoeBackendCapabilities {
  bool supports_full_gpu_resident{true};
  bool supports_hybrid_doorbell_decode{false};
  bool supports_gpu_streaming_prefill{false};
  bool supports_file_backed_bounded_host_cache{false};
  bool supports_cuda_graph_capture{false};
  bool decode_kernel_skips_negative_slots{true};

  size_t source_bytes_per_expert{0};
  size_t gpu_slot_bytes{0};
  size_t gpu_fixed_table_bytes{0};
  size_t cpu_slot_bytes{0};
  size_t min_host_cache_slots{2};
  size_t max_init_conversion_scratch_bytes{16ULL << 20};
};

struct MoeNodeExecutionPlan {
  int32_t layer_index{0};
  int32_t num_experts{0};
  int32_t resident_slots{0};
  int32_t spare_slots{0};
  int32_t dummy_slots{0};
  int32_t host_cache_slots{0};
  uint64_t host_cache_bytes{0};
  bool use_file_backing{false};
  bool enable_hybrid_cpu_decode{false};
  bool enable_streaming_dma_prefill{false};
  bool enable_background_promotion{false};
  uint64_t admitted_gpu_expert_bytes{0};
  uint64_t admitted_host_expert_bytes{0};
  std::vector<int32_t> initial_resident_experts;
  std::vector<int32_t> ranked_experts;
};

class IMoeOffloadNode {
 public:
  virtual ~IMoeOffloadNode() = default;
  virtual const MoeExpertSourceDescriptor& GetSourceDescriptor() const noexcept = 0;
  virtual const MoeBackendCapabilities& GetBackendCapabilities() const noexcept = 0;
  virtual common::Status InitializeAdmittedPlan(const MoeOffloadSessionPolicy& policy,
                                                const MoeNodeExecutionPlan& plan) = 0;
};

}  // namespace onnxruntime::moe_offload
