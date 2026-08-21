import types
import unittest

import numpy as np

from surface_probe_fields import (
    DIFFUSE_IRRADIANCE_RGB_FIELD,
    SurfaceProbeFieldDesc,
    SurfaceProbeFieldSemantic,
    SurfaceProbeFieldStorage,
    SurfaceProbeRuntimeBuffers,
)
from surface_probe_resources import SurfaceProbeGpuGeometry


class _FakeRenderData:
    def __init__(self):
        self.requests = []

    def get_buffer(self, name, **kwargs):
        buffer = object()
        self.requests.append((name, kwargs, buffer))
        return buffer


class _FakeDevice:
    def __init__(self):
        self.requests = []

    def create_buffer(self, **kwargs):
        buffer = object()
        self.requests.append((kwargs, buffer))
        return buffer


class SurfaceProbeFieldTests(unittest.TestCase):
    def test_diffuse_irradiance_field_keeps_existing_working_footprint(self):
        self.assertEqual(DIFFUSE_IRRADIANCE_RGB_FIELD.value_stride, 12)
        self.assertEqual(DIFFUSE_IRRADIANCE_RGB_FIELD.bytes_per_probe, 16)

    def test_invalid_field_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            SurfaceProbeFieldDesc(
                semantic=SurfaceProbeFieldSemantic.DIFFUSE_PRT_L2_SCALAR,
                coefficient_count=0,
                channel_count=1,
                working_storage=SurfaceProbeFieldStorage.FLOAT32,
            )

    def test_prt_l2_field_can_reuse_runtime_without_irradiance_attachments(self):
        prt_desc = SurfaceProbeFieldDesc(
            semantic=SurfaceProbeFieldSemantic.DIFFUSE_PRT_L2_SCALAR,
            coefficient_count=9,
            channel_count=1,
            working_storage=SurfaceProbeFieldStorage.FLOAT32,
        )
        render_data = _FakeRenderData()
        runtime = SurfaceProbeRuntimeBuffers.acquire(
            render_data,
            11,
            field_desc=prt_desc,
            attachment_descs=(),
        )
        self.assertEqual(prt_desc.value_stride, 36)
        self.assertEqual(len(render_data.requests), 2)
        self.assertEqual(runtime.attachments.buffers, {})

    def test_runtime_resources_separate_values_counts_and_attachments(self):
        render_data = _FakeRenderData()
        runtime = SurfaceProbeRuntimeBuffers.acquire(render_data, 7)
        requests = {name: kwargs for name, kwargs, _ in render_data.requests}
        value_request = next(
            kwargs
            for name, kwargs in requests.items()
            if name.endswith(".values")
        )
        count_request = requests[
            "surface_probe_renderer.field.diffuse_irradiance_rgb.v1."
            "sample_counts"
        ]
        self.assertEqual(value_request["size"], 7 * 12)
        self.assertEqual(count_request["struct_size"], 4)
        self.assertEqual(count_request["element_count"], 7)
        self.assertIs(runtime.field.desc, DIFFUSE_IRRADIANCE_RGB_FIELD)
        self.assertIsNot(runtime.field.values, runtime.field.sample_counts)
        self.assertEqual(len(render_data.requests), 4)


class SurfaceProbeGpuGeometryTests(unittest.TestCase):
    def test_geometry_upload_contains_no_field_buffers(self):
        layout = types.SimpleNamespace(
            probes=np.zeros((3, 48), dtype=np.uint8),
            nodes=np.zeros((2, 4), dtype=np.uint32),
            instance_gpu_data=np.zeros((1, 48), dtype=np.uint8),
            triangle_vertex_probes=np.zeros((4, 4), dtype=np.uint32),
            total_probe_count=3,
            instance_infos=(object(),),
        )
        device = _FakeDevice()
        geometry = SurfaceProbeGpuGeometry.create(device, layout)
        labels = {request[0]["label"] for request in device.requests}
        self.assertEqual(geometry.probe_count, 3)
        self.assertEqual(geometry.instance_count, 1)
        self.assertEqual(len(device.requests), 4)
        self.assertEqual(
            labels,
            {
                "surface_probe_metadata",
                "surface_probe_nodes",
                "surface_probe_instances",
                "surface_probe_triangle_vertex_map",
            },
        )


if __name__ == "__main__":
    unittest.main()
