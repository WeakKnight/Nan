"""
BVH2 Builder - CPU-side SAH BVH2 construction for software ray tracing.

Builds a two-level acceleration structure (BLAS + TLAS) compatible with
GPU traversal in software_rt.slang.

BVH Node layout (32 bytes, matches GPU struct):
    float3 aabb_min       (12 bytes)
    uint   left_or_prim   (4 bytes)  - internal: left child index; leaf: first primitive index
    float3 aabb_max       (12 bytes)
    uint   right_or_count (4 bytes)  - internal: right child index; leaf: prim_count | LEAF_FLAG

Leaf nodes are identified by LEAF_FLAG (0x80000000) set in right_or_count.
"""

import numpy as np
import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional


LEAF_FLAG = 0x80000000
SAH_TRAVERSAL_COST = 1.0
SAH_INTERSECTION_COST = 1.0
SAH_NUM_BINS = 12
MAX_LEAF_PRIMS = 4


@dataclass
class AABB:
    """Axis-aligned bounding box."""
    lo: np.ndarray  # (3,) float32
    hi: np.ndarray  # (3,) float32

    @staticmethod
    def empty():
        return AABB(
            lo=np.full(3, np.inf, dtype=np.float32),
            hi=np.full(3, -np.inf, dtype=np.float32),
        )

    def expand_point(self, p: np.ndarray):
        self.lo = np.minimum(self.lo, p)
        self.hi = np.maximum(self.hi, p)

    def expand_aabb(self, other: 'AABB'):
        self.lo = np.minimum(self.lo, other.lo)
        self.hi = np.maximum(self.hi, other.hi)

    def surface_area(self) -> float:
        d = np.maximum(self.hi - self.lo, 0.0)
        return 2.0 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])

    def centroid(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)


class BVHBuilder:
    """
    Builds a BVH2 using SAH (Surface Area Heuristic).
    
    Supports two modes:
    - BLAS: build from triangle primitives (vertices + indices)
    - TLAS: build from instance AABBs
    """

    def __init__(self):
        self.nodes: List[dict] = []  # flat node list

    def build_blas(self, vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Build a BLAS from triangle mesh data.
        
        Args:
            vertices: (V, 8) float32 array [pos.x, pos.y, pos.z, n.x, n.y, n.z, u, v]
                      or (V, 3) float32 positions only
            indices:  (T, 3) uint32 triangle indices
        
        Returns:
            Packed node buffer as (N, 8) float32 / uint32 array (32 bytes per node).
            Also accessible via self.nodes after building.
        """
        self.nodes = []

        num_tris = indices.shape[0]
        if num_tris == 0:
            # Empty mesh: single leaf with 0 primitives
            self.nodes.append({
                'aabb_min': np.zeros(3, dtype=np.float32),
                'aabb_max': np.zeros(3, dtype=np.float32),
                'left': 0,
                'right': LEAF_FLAG | 0,
            })
            return self._pack_nodes()

        # Extract positions (first 3 floats per vertex)
        if vertices.shape[1] > 3:
            positions = vertices[:, :3].astype(np.float32)
        else:
            positions = vertices.astype(np.float32)

        # Compute per-triangle AABBs and centroids
        tri_verts = positions[indices]  # (T, 3, 3)
        tri_aabb_min = tri_verts.min(axis=1)  # (T, 3)
        tri_aabb_max = tri_verts.max(axis=1)  # (T, 3)
        tri_centroids = 0.5 * (tri_aabb_min + tri_aabb_max)  # (T, 3)

        # Primitive indices (will be reordered during build)
        prim_indices = np.arange(num_tris, dtype=np.int32)

        # Build recursively
        self._build_recursive(
            prim_indices, tri_aabb_min, tri_aabb_max, tri_centroids,
            0, num_tris
        )

        # Reorder primitives: return the reordered primitive index mapping
        self._prim_order = prim_indices.copy()

        return self._pack_nodes()

    def build_tlas(self, instance_aabbs: List[AABB]) -> np.ndarray:
        """
        Build a TLAS from instance world-space AABBs.
        
        Args:
            instance_aabbs: List of AABBs, one per instance.
        
        Returns:
            Packed node buffer as bytes.
        """
        self.nodes = []

        num_instances = len(instance_aabbs)
        if num_instances == 0:
            self.nodes.append({
                'aabb_min': np.zeros(3, dtype=np.float32),
                'aabb_max': np.zeros(3, dtype=np.float32),
                'left': 0,
                'right': LEAF_FLAG | 0,
            })
            return self._pack_nodes()

        # Build arrays
        inst_aabb_min = np.array([a.lo for a in instance_aabbs], dtype=np.float32)
        inst_aabb_max = np.array([a.hi for a in instance_aabbs], dtype=np.float32)
        inst_centroids = 0.5 * (inst_aabb_min + inst_aabb_max)

        prim_indices = np.arange(num_instances, dtype=np.int32)

        self._build_recursive(
            prim_indices, inst_aabb_min, inst_aabb_max, inst_centroids,
            0, num_instances
        )

        self._prim_order = prim_indices.copy()
        return self._pack_nodes()

    def get_prim_order(self) -> np.ndarray:
        """Get the reordered primitive indices after build."""
        return self._prim_order

    def _build_recursive(
        self,
        prim_indices: np.ndarray,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
        centroids: np.ndarray,
        start: int,
        end: int,
    ) -> int:
        """
        Recursively build BVH2 with SAH.
        
        Operates on prim_indices[start:end], reordering in place.
        Returns the index of the created node in self.nodes.
        """
        count = end - start
        node_idx = len(self.nodes)
        self.nodes.append(None)  # placeholder

        # Compute bounds for this range
        idxs = prim_indices[start:end]
        node_aabb_min = aabb_min[idxs].min(axis=0)
        node_aabb_max = aabb_max[idxs].max(axis=0)

        # Leaf condition
        if count <= MAX_LEAF_PRIMS:
            self.nodes[node_idx] = {
                'aabb_min': node_aabb_min,
                'aabb_max': node_aabb_max,
                'left': start,
                'right': LEAF_FLAG | count,
            }
            return node_idx

        # Try SAH split along each axis
        best_cost = np.inf
        best_axis = -1
        best_split_pos = start
        parent_sa = max(self._surface_area(node_aabb_min, node_aabb_max), 1e-30)

        cent = centroids[idxs]  # (count, 3)
        cent_min = cent.min(axis=0)
        cent_max = cent.max(axis=0)

        for axis in range(3):
            if cent_max[axis] - cent_min[axis] < 1e-10:
                continue

            # Binned SAH
            bin_min = cent_min[axis]
            bin_max = cent_max[axis]
            bin_scale = SAH_NUM_BINS / (bin_max - bin_min)

            # Assign primitives to bins
            bin_ids = np.clip(
                ((cent[:, axis] - bin_min) * bin_scale).astype(np.int32),
                0, SAH_NUM_BINS - 1
            )

            # Accumulate per-bin bounds
            bin_aabb_min = np.full((SAH_NUM_BINS, 3), np.inf, dtype=np.float32)
            bin_aabb_max = np.full((SAH_NUM_BINS, 3), -np.inf, dtype=np.float32)
            bin_count = np.zeros(SAH_NUM_BINS, dtype=np.int32)

            for i in range(count):
                b = bin_ids[i]
                bin_aabb_min[b] = np.minimum(bin_aabb_min[b], aabb_min[idxs[i]])
                bin_aabb_max[b] = np.maximum(bin_aabb_max[b], aabb_max[idxs[i]])
                bin_count[b] += 1

            # Sweep from left to find costs
            left_aabb_min = np.full(3, np.inf, dtype=np.float32)
            left_aabb_max = np.full(3, -np.inf, dtype=np.float32)
            left_count = 0
            left_costs = np.full(SAH_NUM_BINS - 1, np.inf)
            left_areas = np.zeros(SAH_NUM_BINS - 1)
            left_counts = np.zeros(SAH_NUM_BINS - 1, dtype=np.int32)

            for i in range(SAH_NUM_BINS - 1):
                if bin_count[i] > 0:
                    left_aabb_min = np.minimum(left_aabb_min, bin_aabb_min[i])
                    left_aabb_max = np.maximum(left_aabb_max, bin_aabb_max[i])
                left_count += bin_count[i]
                left_areas[i] = self._surface_area(left_aabb_min, left_aabb_max) if left_count > 0 else 0.0
                left_counts[i] = left_count

            # Sweep from right
            right_aabb_min = np.full(3, np.inf, dtype=np.float32)
            right_aabb_max = np.full(3, -np.inf, dtype=np.float32)
            right_count = 0

            for i in range(SAH_NUM_BINS - 1, 0, -1):
                if bin_count[i] > 0:
                    right_aabb_min = np.minimum(right_aabb_min, bin_aabb_min[i])
                    right_aabb_max = np.maximum(right_aabb_max, bin_aabb_max[i])
                right_count += bin_count[i]
                right_area = self._surface_area(right_aabb_min, right_aabb_max) if right_count > 0 else 0.0
                right_count_val = right_count

                idx = i - 1
                cost = SAH_TRAVERSAL_COST + SAH_INTERSECTION_COST * (
                    left_counts[idx] * left_areas[idx] + right_count_val * right_area
                ) / parent_sa

                if cost < best_cost and left_counts[idx] > 0 and right_count_val > 0:
                    best_cost = cost
                    best_axis = axis
                    best_split_pos = i  # split at bin boundary i

        # Leaf cost
        leaf_cost = SAH_INTERSECTION_COST * count

        if best_axis == -1 or best_cost >= leaf_cost:
            # Make leaf
            if count <= 2 * MAX_LEAF_PRIMS:
                self.nodes[node_idx] = {
                    'aabb_min': node_aabb_min,
                    'aabb_max': node_aabb_max,
                    'left': start,
                    'right': LEAF_FLAG | count,
                }
                return node_idx
            else:
                # Force split at midpoint of largest axis
                extents = cent_max - cent_min
                best_axis = int(np.argmax(extents))
                if extents[best_axis] < 1e-10:
                    # All centroids coincide, split in half
                    mid = start + count // 2
                    left_child = self._build_recursive(
                        prim_indices, aabb_min, aabb_max, centroids, start, mid
                    )
                    right_child = self._build_recursive(
                        prim_indices, aabb_min, aabb_max, centroids, mid, end
                    )
                    self.nodes[node_idx] = {
                        'aabb_min': node_aabb_min,
                        'aabb_max': node_aabb_max,
                        'left': left_child,
                        'right': right_child,
                    }
                    return node_idx

                median = 0.5 * (cent_min[best_axis] + cent_max[best_axis])
                # Partition
                mid = self._partition(prim_indices, centroids, best_axis, median, start, end)
                if mid == start or mid == end:
                    mid = start + count // 2

                left_child = self._build_recursive(
                    prim_indices, aabb_min, aabb_max, centroids, start, mid
                )
                right_child = self._build_recursive(
                    prim_indices, aabb_min, aabb_max, centroids, mid, end
                )
                self.nodes[node_idx] = {
                    'aabb_min': node_aabb_min,
                    'aabb_max': node_aabb_max,
                    'left': left_child,
                    'right': right_child,
                }
                return node_idx

        # Partition by SAH split
        bin_min = cent_min[best_axis]
        bin_max = cent_max[best_axis]
        bin_scale = SAH_NUM_BINS / (bin_max - bin_min)

        split_threshold = bin_min + best_split_pos / bin_scale

        mid = self._partition(prim_indices, centroids, best_axis, split_threshold, start, end)
        if mid == start or mid == end:
            mid = start + count // 2

        left_child = self._build_recursive(
            prim_indices, aabb_min, aabb_max, centroids, start, mid
        )
        right_child = self._build_recursive(
            prim_indices, aabb_min, aabb_max, centroids, mid, end
        )

        self.nodes[node_idx] = {
            'aabb_min': node_aabb_min,
            'aabb_max': node_aabb_max,
            'left': left_child,
            'right': right_child,
        }
        return node_idx

    def _partition(
        self,
        prim_indices: np.ndarray,
        centroids: np.ndarray,
        axis: int,
        threshold: float,
        start: int,
        end: int,
    ) -> int:
        """Partition prim_indices[start:end] by centroid[axis] < threshold. Returns split index."""
        i = start
        j = end - 1
        while i <= j:
            if centroids[prim_indices[i], axis] < threshold:
                i += 1
            else:
                prim_indices[i], prim_indices[j] = prim_indices[j], prim_indices[i]
                j -= 1
        return i

    @staticmethod
    def _surface_area(lo: np.ndarray, hi: np.ndarray) -> float:
        d = np.maximum(hi - lo, 0.0)
        return 2.0 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])

    def _pack_nodes(self) -> np.ndarray:
        """
        Pack nodes into a flat buffer.
        Returns (N, 8) uint32 array where each row is:
            [aabb_min.x_bits, aabb_min.y_bits, aabb_min.z_bits, left,
             aabb_max.x_bits, aabb_max.y_bits, aabb_max.z_bits, right]
        """
        n = len(self.nodes)
        buf = np.zeros((n, 8), dtype=np.uint32)
        for i, node in enumerate(self.nodes):
            lo = node['aabb_min'].astype(np.float32)
            hi = node['aabb_max'].astype(np.float32)
            buf[i, 0] = lo.view(np.uint32)[0]
            buf[i, 1] = lo.view(np.uint32)[1]
            buf[i, 2] = lo.view(np.uint32)[2]
            buf[i, 3] = np.uint32(node['left'])
            buf[i, 4] = hi.view(np.uint32)[0]
            buf[i, 5] = hi.view(np.uint32)[1]
            buf[i, 6] = hi.view(np.uint32)[2]
            buf[i, 7] = np.uint32(node['right'])
        return buf


def build_blas_for_mesh(
    vertices: np.ndarray,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a BLAS for a single mesh.
    
    Args:
        vertices: (V, 8) or (V, 3) float32
        indices:  (T, 3) uint32
    
    Returns:
        (bvh_nodes, prim_order):
            bvh_nodes:  (N, 8) uint32 packed BVH nodes
            prim_order: (T,) int32 reordered triangle indices
    """
    builder = BVHBuilder()
    nodes = builder.build_blas(vertices, indices)
    return nodes, builder.get_prim_order()


def build_tlas_for_instances(
    instance_aabbs: List[AABB],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a TLAS for scene instances.
    
    Args:
        instance_aabbs: World-space AABBs for each instance.
    
    Returns:
        (tlas_nodes, instance_order):
            tlas_nodes:     (N, 8) uint32 packed BVH nodes
            instance_order: (I,) int32 reordered instance indices
    """
    builder = BVHBuilder()
    nodes = builder.build_tlas(instance_aabbs)
    return nodes, builder.get_prim_order()


def compute_mesh_world_aabb(
    vertices: np.ndarray,
    indices: np.ndarray,
    transform: np.ndarray,
) -> AABB:
    """
    Compute world-space AABB for a mesh instance.
    
    Args:
        vertices: (V, 8) or (V, 3) float32
        indices:  (T, 3) uint32
        transform: (4, 4) float32 transform matrix
    
    Returns:
        World-space AABB.
    """
    if vertices.shape[1] > 3:
        positions = vertices[:, :3]
    else:
        positions = vertices

    # Get unique vertex indices used
    unique_idx = np.unique(indices.flatten())
    pts = positions[unique_idx]  # (K, 3)

    # Transform to world space
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)  # (K, 4)
    pts_world = (transform @ pts_h.T).T[:, :3]  # (K, 3)

    aabb = AABB(
        lo=pts_world.min(axis=0).astype(np.float32),
        hi=pts_world.max(axis=0).astype(np.float32),
    )
    return aabb
