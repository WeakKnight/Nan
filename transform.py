import slangpy as spy


class Transform:
    def __init__(self):
        super().__init__()
        self.translation = spy.float3(0)
        self.scaling = spy.float3(1)
        self.rotation = spy.float3(0)
        self.matrix = spy.float4x4.identity()

    def update_matrix(self):
        T = spy.math.matrix_from_translation(self.translation)
        S = spy.math.matrix_from_scaling(self.scaling)
        R = spy.math.matrix_from_rotation_xyz(self.rotation)
        self.matrix = spy.math.mul(spy.math.mul(T, R), S)

