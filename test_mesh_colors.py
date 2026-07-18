import unittest

from mesh_colors import (
    MESH_COLORS_INVALID_OFFSET,
    MESH_COLORS_SIDE_FLAG_DOUBLE_SIDED,
    MeshColorsLayout,
    address_to_ij,
    barycentric_at,
    colors_per_patch,
    ij_to_address,
)
from material import MATERIAL_FLAG_DOUBLE_SIDED, Material
from scene_node import SceneNode


class MeshColorsMathTests(unittest.TestCase):
    def test_address_round_trip(self):
        for resolution in (1, 2, 4, 8, 16):
            for address in range(colors_per_patch(resolution)):
                i, j = address_to_ij(address, resolution)
                self.assertEqual(ij_to_address(i, j, resolution), address)
                barycentrics = barycentric_at(i, j, resolution)
                self.assertAlmostEqual(sum(barycentrics), 1.0)
                self.assertTrue(all(value >= 0.0 for value in barycentrics))

    def test_demo_layout_offsets_and_resolution_clamp(self):
        scene_node = SceneNode.demo()
        layout = MeshColorsLayout.build(
            scene_node,
            texels_per_unit=64.0,
            min_resolution=2,
            max_resolution=8,
            max_total_texels=100_000,
        )
        self.assertGreater(layout.total_texel_count, 0)
        self.assertEqual(len(layout.instance_infos), len(scene_node.instances))
        self.assertEqual(
            len(layout.face_infos),
            sum(
                scene_node.meshes[mesh_id].triangle_count
                for mesh_id, _, _ in scene_node.instances
            ),
        )

        expected_payload_offset = 0
        expected_surface_texels = 0
        allocated_ranges = []
        for instance, side in zip(layout.instance_infos, layout.side_infos):
            self.assertEqual(instance.texel_offset, expected_payload_offset)
            expected_local_address = 0
            for face_index in range(instance.face_count):
                face = layout.face_infos[instance.face_offset + face_index]
                self.assertEqual(face.address, expected_local_address)
                self.assertGreaterEqual(face.resolution, 2)
                self.assertLessEqual(face.resolution, 8)
                expected_local_address += colors_per_patch(face.resolution)
            self.assertEqual(instance.texel_count, expected_local_address)
            front_range = (
                instance.texel_offset,
                instance.texel_offset + instance.texel_count,
            )
            allocated_ranges.append(front_range)
            if side.is_double_sided:
                self.assertEqual(side.back_texel_offset, front_range[1])
                allocated_ranges.append(
                    (
                        side.back_texel_offset,
                        side.back_texel_offset + instance.texel_count,
                    )
                )
                expected_payload_offset += instance.texel_count * 2
            else:
                self.assertEqual(
                    side.back_texel_offset, MESH_COLORS_INVALID_OFFSET
                )
                expected_payload_offset += instance.texel_count
            expected_surface_texels += instance.texel_count

        for previous, current in zip(allocated_ranges, allocated_ranges[1:]):
            self.assertEqual(previous[1], current[0])
        self.assertEqual(expected_surface_texels, layout.total_surface_texels)
        self.assertEqual(expected_payload_offset, layout.total_payload_count)

    def test_material_flags_and_cornell_material_roles(self):
        self.assertEqual(Material().flags, 0)
        double_sided = Material(double_sided=True)
        self.assertTrue(double_sided.double_sided)
        self.assertEqual(
            double_sided.flags & MATERIAL_FLAG_DOUBLE_SIDED,
            MATERIAL_FLAG_DOUBLE_SIDED,
        )

        scene_node = SceneNode.demo()
        for instance_index, (_, material_id, _) in enumerate(scene_node.instances):
            material = scene_node.materials[material_id]
            if instance_index < 5:
                self.assertTrue(material.double_sided)
                self.assertNotEqual(
                    material.flags & MATERIAL_FLAG_DOUBLE_SIDED, 0
                )
            else:
                self.assertFalse(material.double_sided)
                self.assertEqual(
                    material.flags & MATERIAL_FLAG_DOUBLE_SIDED, 0
                )

    def test_selective_back_payload_count_and_budget(self):
        scene_node = SceneNode.demo()
        layout = MeshColorsLayout.build(
            scene_node,
            texels_per_unit=1.0,
            min_resolution=2,
            max_resolution=2,
            max_total_texels=276,
        )
        self.assertEqual(layout.total_surface_texels, 216)
        self.assertEqual(layout.total_payload_count, 276)
        self.assertEqual(
            sum(
                instance.texel_count
                for instance, side in zip(
                    layout.instance_infos, layout.side_infos
                )
                if side.flags & MESH_COLORS_SIDE_FLAG_DOUBLE_SIDED
            ),
            60,
        )

        with self.assertRaisesRegex(ValueError, "texel budget exceeded"):
            MeshColorsLayout.build(
                scene_node,
                texels_per_unit=1.0,
                min_resolution=2,
                max_resolution=2,
                max_total_texels=275,
            )

    def test_budget_guard_rejects_impossible_layout(self):
        with self.assertRaisesRegex(ValueError, "texel budget exceeded"):
            MeshColorsLayout.build(
                SceneNode.demo(),
                min_resolution=4,
                max_resolution=4,
                max_total_texels=1,
            )

    def test_resolution_limits_normalize_inward(self):
        layout = MeshColorsLayout.build(
            SceneNode.demo(),
            min_resolution=3,
            max_resolution=7,
            max_total_texels=100_000,
        )
        self.assertEqual(layout.min_resolution, 4)
        self.assertEqual(layout.max_resolution, 4)

        with self.assertRaisesRegex(ValueError, "min_resolution"):
            MeshColorsLayout.build(
                SceneNode.demo(),
                min_resolution=5,
                max_resolution=7,
                max_total_texels=100_000,
            )


if __name__ == "__main__":
    unittest.main()
