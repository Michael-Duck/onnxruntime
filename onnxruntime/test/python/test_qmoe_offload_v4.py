#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Unit tests for Portable QMoE Offload v4 (standard com.microsoft::QMoE + session.moe.offload.*)."""

import contextlib
import ctypes
import os
import struct
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort


def _float32_to_bfloat16_u16(arr: np.ndarray) -> np.ndarray:
    u32 = arr.astype(np.float32).view(np.uint32)
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)


def _bfloat16_u16_to_float32(arr: np.ndarray) -> np.ndarray:
    u32 = arr.astype(np.uint32) << 16
    return u32.view(np.float32)


def _build_canonical_nvfp4_qmoe_model(
    model_dir: str,
    num_layers: int = 2,
    num_experts: int = 16,
    hidden_size: int = 128,
    inter_size: int = 64,
    top_k: int = 4,
) -> str:
    rng = np.random.default_rng(42)
    ext_filename = "weights.data"
    ext_path = os.path.join(model_dir, ext_filename)

    nodes = []
    initializers = []
    offset = 0

    with open(ext_path, "wb") as wf:
        cur_in = "X"
        for l_idx in range(num_layers):
            fc1_w = rng.integers(0, 256, size=(num_experts, hidden_size, inter_size), dtype=np.uint8)
            fc2_w = rng.integers(0, 256, size=(num_experts, inter_size, hidden_size // 2), dtype=np.uint8)
            # Use mild E4M3 scale bytes (0x38 = 0.5, 0x30 = 0.25) so outputs stay well-conditioned
            fc1_s = rng.choice(
                np.array([0x30, 0x34, 0x38], dtype=np.uint8), size=(num_experts, 2 * inter_size, hidden_size // 16)
            )
            fc2_s = rng.choice(
                np.array([0x30, 0x34, 0x38], dtype=np.uint8), size=(num_experts, hidden_size, inter_size // 16)
            )
            fc1_g = np.full((num_experts,), 0.05, dtype=np.float32)
            fc2_g = np.full((num_experts,), 0.05, dtype=np.float32)

            def add_ext_tensor(name, arr, onnx_dtype):
                nonlocal offset
                raw = arr.tobytes()
                pad = (4096 - (offset % 4096)) % 4096
                if pad:
                    wf.write(b"\x00" * pad)
                    offset += pad
                wf.write(raw)
                t = helper.make_tensor(
                    name=name,
                    data_type=onnx_dtype,
                    dims=list(arr.shape),
                    vals=raw,
                    raw=True,
                )
                onnx.external_data_helper.set_external_data(t, location=ext_filename, offset=offset, length=len(raw))
                t.data_location = TensorProto.EXTERNAL
                t.ClearField("raw_data")
                offset += len(raw)
                initializers.append(t)

            w1_name = f"model.layers.{l_idx}.mlp.fc1_experts_weights"
            s1_name = f"model.layers.{l_idx}.mlp.fc1_scales"
            w2_name = f"model.layers.{l_idx}.mlp.fc2_experts_weights"
            s2_name = f"model.layers.{l_idx}.mlp.fc2_scales"
            g1_name = f"model.layers.{l_idx}.mlp.fc1_global_scales"
            g2_name = f"model.layers.{l_idx}.mlp.fc2_global_scales"

            add_ext_tensor(w1_name, fc1_w, TensorProto.UINT8)
            add_ext_tensor(s1_name, fc1_s, TensorProto.FLOAT8E4M3FN)
            add_ext_tensor(w2_name, fc2_w, TensorProto.UINT8)
            add_ext_tensor(s2_name, fc2_s, TensorProto.FLOAT8E4M3FN)
            add_ext_tensor(g1_name, fc1_g, TensorProto.FLOAT)
            add_ext_tensor(g2_name, fc2_g, TensorProto.FLOAT)

            out_name = "Y" if l_idx == num_layers - 1 else f"H_{l_idx}"
            node = helper.make_node(
                "QMoE",
                inputs=[
                    cur_in,
                    "router_probs",
                    w1_name,
                    s1_name,
                    "",
                    w2_name,
                    s2_name,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    g1_name,
                    g2_name,
                ],
                outputs=[out_name],
                name=f"model/layers.{l_idx}/mlp/QMoE_node_{l_idx}",
                domain="com.microsoft",
                activation_type="swiglu",
                normalize_routing_weights=1,
                k=top_k,
                expert_weight_bits=4,
                block_size=16,
                swiglu_fusion=1,
                quant_type="nvfp4",
                weights_prepacked=1,
            )
            nodes.append(node)
            cur_in = out_name

    # Also write a STRP v1 popularity profile (expert-profile.bin)
    prof_path = os.path.join(model_dir, "expert-profile.bin")
    pairs = []
    for e in range(num_experts):
        for l_idx in range(num_layers):
            pairs.append((l_idx, (e * 3 + l_idx) % num_experts))
    with open(prof_path, "wb") as pf:
        pf.write(struct.pack("<6I", 0x50525453, 1, num_layers, num_experts, len(pairs), len(pairs)))
        pf.writelines(struct.pack("<2H", l_id, e_id) for l_id, e_id in pairs)

    graph = helper.make_graph(
        nodes,
        "qmoe_v4_test",
        [
            helper.make_tensor_value_info("X", TensorProto.BFLOAT16, ["T", hidden_size]),
            helper.make_tensor_value_info("router_probs", TensorProto.BFLOAT16, ["T", num_experts]),
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, ["T", hidden_size]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 20),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model_path = os.path.join(model_dir, "model.onnx")
    onnx.save_model(model, model_path)
    return model_path


class TestQMoEOffloadV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin_path = os.environ.get("QWEN_PROVIDER")
        if plugin_path and os.path.exists(plugin_path):
            with contextlib.suppress(Exception):
                ort.register_execution_provider_library("CUDAExecutionProvider", plugin_path)

    def _make_session(self, model_path: str, config_entries: dict, cuda_options: dict | None = None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        for k, v in config_entries.items():
            so.add_session_config_entry(k, str(v))
        providers = [("CUDAExecutionProvider", cuda_options or {})]
        return ort.InferenceSession(model_path, sess_options=so, providers=providers)

    def _run_bf16(self, sess, x_f32: np.ndarray, r_f32: np.ndarray) -> np.ndarray:
        x_bf16 = _float32_to_bfloat16_u16(x_f32)
        r_bf16 = _float32_to_bfloat16_u16(r_f32)
        ort_x = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(x_bf16, TensorProto.BFLOAT16)
        ort_r = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r_bf16, TensorProto.BFLOAT16)
        out = sess.run_with_ort_values(["Y"], {"X": ort_x, "router_probs": ort_r})[0]
        raw_bytes = bytes((ctypes.c_uint16 * int(np.prod(x_f32.shape))).from_address(out.data_ptr()))
        return _bfloat16_u16_to_float32(np.frombuffer(raw_bytes, dtype=np.uint16).reshape(x_f32.shape))

    def test_discarded_prototype_features_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(tmpdir)
            # 1. Unknown session.moe.offload.* key must fail with INVALID_ARGUMENT
            with self.assertRaisesRegex(Exception, "Unknown session.moe.offload"):
                self._make_session(model_path, {"session.moe.offload.unknown_knob": "1"})

            # 2. Discarded prototype moe.cpu_offload.* key must fail with INVALID_ARGUMENT
            with self.assertRaisesRegex(Exception, "discarded and not supported"):
                self._make_session(model_path, {"moe.cpu_offload.experts": "4"})

            # 3. Discarded prototype placement strategy 'basic' must fail with INVALID_ARGUMENT
            with self.assertRaisesRegex(Exception, "expected 'static' or 'adaptive'"):
                self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "basic",
                    },
                )

            # 4. Separate QMoEOffload operator schema must no longer exist
            m = onnx.load_model(model_path, load_external_data=False)
            m.graph.node[0].op_type = "QMoEOffload"
            legacy_path = os.path.join(tmpdir, "legacy_qmoe_offload.onnx")
            onnx.save_model(m, legacy_path)
            with self.assertRaisesRegex(Exception, "QMoEOffload"):
                self._make_session(legacy_path, {})

    def test_v4_portable_qmoe_modes_and_numerical_parity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir, num_layers=2, num_experts=16, hidden_size=128, inter_size=64, top_k=4
            )
            # Per-expert slot bytes for H=128, I=64, block=16:
            # 2*64*64 + 128*32 + 2*64*8 + 128*4 + 8 = 8192 + 4096 + 1024 + 512 + 8 = 13,832 B
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4  # 128 B per layer

            # 1. Baseline: mode=disabled (Full GPU resident)
            sess_base = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})

            # 2. mode=auto with ample VRAM (Full GPU residency admitted via Stage 1->4)
            sess_auto_full = self._make_session(model_path, {"session.moe.offload.mode": "auto"})

            # 3. mode=required, static placement (6 resident slots/layer = 12 total slots across 2 layers, spare=0)
            budget_12_slots = 2 * fixed_table_bytes + 12 * slot_bytes
            sess_static = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(budget_12_slots),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )

            # 4. mode=required, adaptive placement (6 resident + 2 spare slots/layer = 16 total physical slots)
            budget_adaptive = 2 * fixed_table_bytes + 16 * slot_bytes
            sess_adaptive = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )

            # 5. mode=required with bounded host_expert_bytes (FileExpertStore bounded host cache)
            sess_bounded_host = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.host_expert_bytes": str(2 * 4 * slot_bytes),  # 4 cold slots/layer
                    "session.moe.offload.allow_file_backing": "1",
                },
            )

            rng = np.random.default_rng(123)
            # Test both decode regime (T=2 <= async_max_tokens=4) and prefill regime (T=16 > async_max_tokens=4)
            for num_rows in [1, 2, 4, 16]:
                x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 16), dtype=np.float32)

                y_base_f32 = self._run_bf16(sess_base, x, r)

                for name, s in [
                    ("auto_full", sess_auto_full),
                    ("static_hybrid", sess_static),
                    ("adaptive_hybrid", sess_adaptive),
                    ("bounded_host", sess_bounded_host),
                ]:
                    y_cand_f32 = self._run_bf16(s, x, r)
                    max_diff = float(np.max(np.abs(y_base_f32 - y_cand_f32)))
                    self.assertLess(
                        max_diff, 5e-2, f"Numerical mismatch in mode={name} at T={num_rows}: max_diff={max_diff}"
                    )

    def test_v4_cuda_graph_capture_and_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir, num_layers=2, num_experts=16, hidden_size=128, inter_size=64, top_k=4
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            budget_adaptive = 2 * fixed_table_bytes + 16 * slot_bytes

            rng = np.random.default_rng(999)
            x = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
            r = rng.standard_normal((2, 16), dtype=np.float32)
            sess_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})
            y_ref = self._run_bf16(sess_ref, x, r)
            del sess_ref

            # Wrap inputs/outputs in FLOAT <-> BFLOAT16 Cast on GPU so CUDA Graph capture has 100% GPU I/O
            m = onnx.load_model(model_path, load_external_data=False)
            m.graph.node.insert(
                0, helper.make_node("Cast", inputs=["R_f32"], outputs=["router_probs"], to=TensorProto.BFLOAT16)
            )
            m.graph.node.insert(0, helper.make_node("Cast", inputs=["X_f32"], outputs=["X"], to=TensorProto.BFLOAT16))
            m.graph.node.append(helper.make_node("Cast", inputs=["Y"], outputs=["Y_f32"], to=TensorProto.FLOAT))
            del m.graph.input[:]
            m.graph.input.extend(
                [
                    helper.make_tensor_value_info("X_f32", TensorProto.FLOAT, ["T", 128]),
                    helper.make_tensor_value_info("R_f32", TensorProto.FLOAT, ["T", 16]),
                ]
            )
            del m.graph.output[:]
            m.graph.output.append(helper.make_tensor_value_info("Y_f32", TensorProto.FLOAT, ["T", 128]))
            cg_model_path = os.path.join(tmpdir, "model_cg.onnx")
            onnx.save_model(m, cg_model_path)

            devices = [
                d
                for d in ort.get_ep_devices()
                if d.ep_name == "CUDAExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.GPU
            ]
            devices.sort(key=lambda d: int(d.ep_options.get("device_id", d.ep_metadata.get("cuda_device_id", "0"))))
            dev_id = int(devices[0].ep_options.get("device_id", devices[0].ep_metadata.get("cuda_device_id", "0")))

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            for k, v in {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "adaptive",
                "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                "session.moe.offload.popularity_file": "expert-profile.bin",
            }.items():
                so.add_session_config_entry(k, v)
            so.add_provider_for_devices(
                [devices[0]], {"enable_cuda_graph": "1", "min_num_runs_before_cuda_graph_capture": "2"}
            )
            sess_graph = ort.InferenceSession(cg_model_path, sess_options=so)

            x_in = _bfloat16_u16_to_float32(_float32_to_bfloat16_u16(x))
            r_in = _bfloat16_u16_to_float32(_float32_to_bfloat16_u16(r))
            binding = sess_graph.io_binding()
            binding.bind_cpu_input("X_f32", x_in)
            binding.bind_cpu_input("R_f32", r_in)
            binding.bind_output("Y_f32", "cuda", dev_id)

            for _ in range(6):
                binding.synchronize_inputs()
                sess_graph.run_with_iobinding(binding)
                y_cand = binding.copy_outputs_to_cpu()[0]
                max_diff = float(np.max(np.abs(y_ref - y_cand)))
                self.assertLess(max_diff, 5e-2, f"CUDA graph replay mismatch: max_diff={max_diff}")


if __name__ == "__main__":
    unittest.main()
