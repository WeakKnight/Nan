import slangpy as spy
import numpy as np
import numpy.typing as npt


class Mesh:
    def __init__(
        self,
        vertices: npt.NDArray[np.float32],  # type: ignore
        indices: npt.NDArray[np.uint32],  # type: ignore
    ):
        super().__init__()
        assert vertices.ndim == 2 and vertices.dtype == np.float32
        assert indices.ndim == 2 and indices.dtype == np.uint32
        self.vertices = vertices
        self.indices = indices

    @property
    def vertex_count(self):
        return self.vertices.shape[0]

    @property
    def triangle_count(self):
        return self.indices.shape[0]

    @property
    def index_count(self):
        return self.triangle_count * 3

    @classmethod
    def create_triangle(
        cls,
        pos_a: npt.NDArray[np.float32],  # type: ignore
        pos_b: npt.NDArray[np.float32],  # type: ignore
        pos_c: npt.NDArray[np.float32],  # type: ignore
        normal_a: npt.NDArray[np.float32],  # type: ignore
        normal_b: npt.NDArray[np.float32],  # type: ignore
        normal_c: npt.NDArray[np.float32],  # type: ignore
    ):
        """
        Create a triangle mesh with specified positions and normals for each vertex.
        
        Args:
            pos_a, pos_b, pos_c: 3D positions of vertices A, B, C
            normal_a, normal_b, normal_c: 3D normals at vertices A, B, C
        """
        vertices = np.array(
            [
                # position, normal, uv
                [pos_a[0], pos_a[1], pos_a[2], normal_a[0], normal_a[1], normal_a[2], 0, 0],
                [pos_b[0], pos_b[1], pos_b[2], normal_b[0], normal_b[1], normal_b[2], 1, 0],
                [pos_c[0], pos_c[1], pos_c[2], normal_c[0], normal_c[1], normal_c[2], 0, 1],
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                [0, 1, 2],
            ],
            dtype=np.uint32,
        )
        return Mesh(vertices, indices)

    @classmethod
    def create_quad(cls, size: "spy.float2param" = spy.float2(1)):
        vertices = np.array(
            [
                # position, normal, uv
                [-0.5, 0, -0.5, 0, 1, 0, 0, 0],
                [+0.5, 0, -0.5, 0, 1, 0, 1, 0],
                [-0.5, 0, +0.5, 0, 1, 0, 0, 1],
                [+0.5, 0, +0.5, 0, 1, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        vertices[:, (0, 2)] *= [size[0], size[1]]
        indices = np.array(
            [
                [2, 1, 0],
                [1, 2, 3],
            ],
            dtype=np.uint32,
        )
        return Mesh(vertices, indices)

    @classmethod
    def create_quad_yz(cls, size: "spy.float2param" = spy.float2(1), face_positive_x: bool = True):
        """Create a quad in YZ plane. If face_positive_x=True, normal points +X, else -X."""
        nx = 1.0 if face_positive_x else -1.0
        vertices = np.array(
            [
                # position (x=0), normal, uv
                [0, -0.5, -0.5, nx, 0, 0, 0, 0],
                [0, +0.5, -0.5, nx, 0, 0, 1, 0],
                [0, -0.5, +0.5, nx, 0, 0, 0, 1],
                [0, +0.5, +0.5, nx, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
        vertices[:, (1, 2)] *= [size[0], size[1]]
        if face_positive_x:
            # For normal +X, winding should give cross product pointing +X
            indices = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.uint32)
        else:
            # For normal -X
            indices = np.array([[2, 1, 0], [1, 2, 3]], dtype=np.uint32)
        return Mesh(vertices, indices)

    @classmethod
    def create_quad_xy(cls, size: "spy.float2param" = spy.float2(1), face_positive_z: bool = True):
        """Create a quad in XY plane. If face_positive_z=True, normal points +Z, else -Z."""
        nz = 1.0 if face_positive_z else -1.0
        vertices = np.array(
            [
                # position (z=0), normal, uv
                [-0.5, -0.5, 0, 0, 0, nz, 0, 0],
                [+0.5, -0.5, 0, 0, 0, nz, 1, 0],
                [-0.5, +0.5, 0, 0, 0, nz, 0, 1],
                [+0.5, +0.5, 0, 0, 0, nz, 1, 1],
            ],
            dtype=np.float32,
        )
        vertices[:, (0, 1)] *= [size[0], size[1]]
        if face_positive_z:
            indices = np.array([[0, 1, 2], [3, 2, 1]], dtype=np.uint32)
        else:
            indices = np.array([[2, 1, 0], [1, 2, 3]], dtype=np.uint32)
        return Mesh(vertices, indices)

    @classmethod
    def create_cube(cls, size: "spy.float3param" = spy.float3(1)):
        vertices = np.array(
            [
                # position, normal, uv
                # left
                [-0.5, -0.5, -0.5, 0, -1, 0, 0.0, 0.0],
                [-0.5, -0.5, +0.5, 0, -1, 0, 1.0, 0.0],
                [+0.5, -0.5, +0.5, 0, -1, 0, 1.0, 1.0],
                [+0.5, -0.5, -0.5, 0, -1, 0, 0.0, 1.0],
                # right
                [-0.5, +0.5, +0.5, 0, +1, 0, 0.0, 0.0],
                [-0.5, +0.5, -0.5, 0, +1, 0, 1.0, 0.0],
                [+0.5, +0.5, -0.5, 0, +1, 0, 1.0, 1.0],
                [+0.5, +0.5, +0.5, 0, +1, 0, 0.0, 1.0],
                # back
                [-0.5, +0.5, -0.5, 0, 0, -1, 0.0, 0.0],
                [-0.5, -0.5, -0.5, 0, 0, -1, 1.0, 0.0],
                [+0.5, -0.5, -0.5, 0, 0, -1, 1.0, 1.0],
                [+0.5, +0.5, -0.5, 0, 0, -1, 0.0, 1.0],
                # front
                [+0.5, +0.5, +0.5, 0, 0, +1, 0.0, 0.0],
                [+0.5, -0.5, +0.5, 0, 0, +1, 1.0, 0.0],
                [-0.5, -0.5, +0.5, 0, 0, +1, 1.0, 1.0],
                [-0.5, +0.5, +0.5, 0, 0, +1, 0.0, 1.0],
                # bottom
                [-0.5, +0.5, +0.5, -1, 0, 0, 0.0, 0.0],
                [-0.5, -0.5, +0.5, -1, 0, 0, 1.0, 0.0],
                [-0.5, -0.5, -0.5, -1, 0, 0, 1.0, 1.0],
                [-0.5, +0.5, -0.5, -1, 0, 0, 0.0, 1.0],
                # top
                [+0.5, +0.5, -0.5, +1, 0, 0, 0.0, 0.0],
                [+0.5, -0.5, -0.5, +1, 0, 0, 1.0, 0.0],
                [+0.5, -0.5, +0.5, +1, 0, 0, 1.0, 1.0],
                [+0.5, +0.5, +0.5, +1, 0, 0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        vertices[:, 0:3] *= [size[0], size[1], size[2]]

        indices = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 6, 5],
                [4, 7, 6],
                [8, 10, 9],
                [8, 11, 10],
                [12, 14, 13],
                [12, 15, 14],
                [16, 18, 17],
                [16, 19, 18],
                [20, 22, 21],
                [20, 23, 22],
            ],
            dtype=np.uint32,
        )

        return Mesh(vertices, indices)

