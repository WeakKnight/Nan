#include "vertex_baking_utils.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <string>
#include <vector>

#ifdef VBAKE_USE_EIGEN
#include <Eigen/SparseCholesky>
#endif

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& message)
{
    g_last_error = message;
}

bool finite_float(float value)
{
    return std::isfinite(static_cast<double>(value));
}

bool validate_inputs(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const unsigned int* indices,
    int sample_count,
    const unsigned int* sample_triangles,
    const float* sample_barycentrics,
    const float* sample_values,
    int channels,
    float* out_vertex_values)
{
    if (vertex_count <= 0) {
        set_error("vertex_count must be positive");
        return false;
    }
    if (triangle_count <= 0) {
        set_error("triangle_count must be positive");
        return false;
    }
    if (sample_count <= 0) {
        set_error("sample_count must be positive");
        return false;
    }
    if (channels <= 0 || channels > 16) {
        set_error("channels must be in [1, 16]");
        return false;
    }
    if (!positions || !indices || !sample_triangles || !sample_barycentrics || !sample_values || !out_vertex_values) {
        set_error("null input/output pointer");
        return false;
    }
    return true;
}

struct Vec2 {
    double x;
    double y;
};

struct Vec3 {
    double x;
    double y;
    double z;
};

struct Butterfly {
    int wing_first = -1;
    int wing_second = -1;
    int count = 0;
};

struct TripletEntry {
    int row;
    int col;
    double value;
};

struct SparseMatrixRows {
    std::vector<std::vector<std::pair<int, double>>> rows;
    std::vector<double> diagonal;
};

Vec3 load_position(const float* positions, int index)
{
    return {
        static_cast<double>(positions[static_cast<size_t>(index) * 3 + 0]),
        static_cast<double>(positions[static_cast<size_t>(index) * 3 + 1]),
        static_cast<double>(positions[static_cast<size_t>(index) * 3 + 2]),
    };
}

Vec3 sub(Vec3 a, Vec3 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross(Vec3 a, Vec3 b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double length(Vec3 v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

double triangle_area(Vec3 a, Vec3 b, Vec3 c)
{
    return 0.5 * length(cross(sub(b, a), sub(c, a)));
}

bool planarize_triangle(const Vec3 v[3], Vec2 p[3])
{
    const double l01 = length(sub(v[0], v[1]));
    const double l02 = length(sub(v[0], v[2]));
    const double l12 = length(sub(v[1], v[2]));
    const double eps = 1e-14;
    if (l01 <= eps || l02 <= eps || l12 <= eps) {
        return false;
    }

    const double p2y = (l02 * l02 + l01 * l01 - l12 * l12) / (2.0 * l01);
    const double x_sq = l02 * l02 - p2y * p2y;
    if (x_sq <= eps) {
        return false;
    }

    p[0] = {0.0, 0.0};
    p[1] = {0.0, l01};
    p[2] = {std::sqrt(x_sq), p2y};
    return true;
}

bool tri_grad_2d(const Vec2 p[3], double grad[2][3])
{
    const double det =
        -p[0].y * p[1].x + p[0].x * p[1].y + p[0].y * p[2].x
        - p[1].y * p[2].x - p[0].x * p[2].y + p[1].x * p[2].y;
    if (std::abs(det) <= 1e-14) {
        return false;
    }

    grad[0][0] = p[1].y - p[2].y;
    grad[0][1] = p[2].y - p[0].y;
    grad[0][2] = p[0].y - p[1].y;

    grad[1][0] = p[2].x - p[1].x;
    grad[1][1] = p[0].x - p[2].x;
    grad[1][2] = p[1].x - p[0].x;

    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) {
            grad[row][col] /= det;
        }
    }
    return true;
}

bool butterfly_grad_diff(const Vec3 v[4], double gd[2][4])
{
    Vec3 tri_a[3] = {v[0], v[1], v[2]};
    Vec3 tri_b[3] = {v[0], v[1], v[3]};
    Vec2 p_a[3];
    Vec2 p_b[3];
    if (!planarize_triangle(tri_a, p_a) || !planarize_triangle(tri_b, p_b)) {
        return false;
    }
    p_b[2].x *= -1.0;

    double grad_a[2][3];
    double grad_b[2][3];
    if (!tri_grad_2d(p_a, grad_a) || !tri_grad_2d(p_b, grad_b)) {
        return false;
    }

    for (int row = 0; row < 2; ++row) {
        gd[row][0] = grad_a[row][0] - grad_b[row][0];
        gd[row][1] = grad_a[row][1] - grad_b[row][1];
        gd[row][2] = grad_a[row][2];
        gd[row][3] = -grad_b[row][2];
    }

    const double area = triangle_area(tri_a[0], tri_a[1], tri_a[2]) + triangle_area(tri_b[0], tri_b[1], tri_b[2]);
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 4; ++col) {
            gd[row][col] *= area;
        }
    }
    return true;
}

void add_edge_regularization(
    std::vector<double>& matrix,
    int vertex_count,
    int triangle_count,
    const float* positions,
    const unsigned int* indices,
    double weight)
{
    std::map<std::pair<int, int>, Butterfly> edges;
    for (int tri = 0; tri < triangle_count; ++tri) {
        const int tri_indices[3] = {
            static_cast<int>(indices[static_cast<size_t>(tri) * 3 + 0]),
            static_cast<int>(indices[static_cast<size_t>(tri) * 3 + 1]),
            static_cast<int>(indices[static_cast<size_t>(tri) * 3 + 2]),
        };
        for (int k = 0; k < 3; ++k) {
            const int current = tri_indices[k];
            const int next = tri_indices[(k + 1) % 3];
            const int opposite = tri_indices[(k + 2) % 3];
            const int edge_a = std::min(current, next);
            const int edge_b = std::max(current, next);
            Butterfly& butterfly = edges[std::make_pair(edge_a, edge_b)];
            if (edge_a == current) {
                butterfly.wing_first = opposite;
            } else {
                butterfly.wing_second = opposite;
            }
            butterfly.count++;
        }
    }

    for (const auto& entry : edges) {
        const Butterfly& butterfly = entry.second;
        if (butterfly.count != 2 || butterfly.wing_first < 0 || butterfly.wing_second < 0) {
            continue;
        }

        const int vertex_indices[4] = {
            entry.first.first,
            entry.first.second,
            butterfly.wing_first,
            butterfly.wing_second,
        };
        Vec3 butterfly_vertices[4];
        bool valid = true;
        for (int i = 0; i < 4; ++i) {
            if (vertex_indices[i] < 0 || vertex_indices[i] >= vertex_count) {
                valid = false;
                break;
            }
            butterfly_vertices[i] = load_position(positions, vertex_indices[i]);
        }
        if (!valid) {
            continue;
        }

        double gd[2][4];
        if (!butterfly_grad_diff(butterfly_vertices, gd)) {
            continue;
        }

        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                double value = 0.0;
                for (int k = 0; k < 2; ++k) {
                    value += gd[k][row] * gd[k][col];
                }
                matrix[static_cast<size_t>(vertex_indices[row]) * vertex_count + vertex_indices[col]] += weight * value;
            }
        }
    }
}

bool cholesky_decompose(std::vector<double>& matrix, int n)
{
    const double eps = 1e-12;
    for (int col = 0; col < n; ++col) {
        for (int row = col; row < n; ++row) {
            double sum = matrix[static_cast<size_t>(row) * n + col];
            for (int k = 0; k < col; ++k) {
                sum -= matrix[static_cast<size_t>(row) * n + k] * matrix[static_cast<size_t>(col) * n + k];
            }

            if (row == col) {
                if (!std::isfinite(sum) || sum <= eps) {
                    return false;
                }
                matrix[static_cast<size_t>(row) * n + col] = std::sqrt(sum);
            } else {
                matrix[static_cast<size_t>(row) * n + col] = sum / matrix[static_cast<size_t>(col) * n + col];
            }
        }
    }

    for (int row = 0; row < n; ++row) {
        for (int col = row + 1; col < n; ++col) {
            matrix[static_cast<size_t>(row) * n + col] = 0.0;
        }
    }
    return true;
}

bool cholesky_solve(const std::vector<double>& lower, const std::vector<double>& rhs, int n, int channels, std::vector<double>& x)
{
    std::vector<double> y(static_cast<size_t>(n) * channels, 0.0);

    for (int row = 0; row < n; ++row) {
        const double diag = lower[static_cast<size_t>(row) * n + row];
        if (!std::isfinite(diag) || diag == 0.0) {
            return false;
        }
        for (int c = 0; c < channels; ++c) {
            double sum = rhs[static_cast<size_t>(row) * channels + c];
            for (int k = 0; k < row; ++k) {
                sum -= lower[static_cast<size_t>(row) * n + k] * y[static_cast<size_t>(k) * channels + c];
            }
            y[static_cast<size_t>(row) * channels + c] = sum / diag;
        }
    }

    for (int row = n - 1; row >= 0; --row) {
        const double diag = lower[static_cast<size_t>(row) * n + row];
        if (!std::isfinite(diag) || diag == 0.0) {
            return false;
        }
        for (int c = 0; c < channels; ++c) {
            double sum = y[static_cast<size_t>(row) * channels + c];
            for (int k = row + 1; k < n; ++k) {
                sum -= lower[static_cast<size_t>(k) * n + row] * x[static_cast<size_t>(k) * channels + c];
            }
            x[static_cast<size_t>(row) * channels + c] = sum / diag;
        }
    }
    return true;
}

bool solve_system(std::vector<double>& matrix, const std::vector<double>& rhs, int n, int channels, std::vector<double>& x)
{
#ifdef VBAKE_USE_EIGEN
    typedef Eigen::SparseMatrix<double> SparseMatrix;
    typedef Eigen::Triplet<double> Triplet;

    std::vector<Triplet> triplets;
    triplets.reserve(static_cast<size_t>(n) * 9);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const double value = matrix[static_cast<size_t>(row) * n + col];
            if (value != 0.0) {
                triplets.emplace_back(row, col, value);
            }
        }
    }

    SparseMatrix sparse(n, n);
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    solver.compute(sparse);
    if (solver.info() != Eigen::Success) {
        set_error("Eigen SimplicialLDLT decomposition failed");
        return false;
    }

    for (int c = 0; c < channels; ++c) {
        Eigen::VectorXd b(n);
        for (int row = 0; row < n; ++row) {
            b(row) = rhs[static_cast<size_t>(row) * channels + c];
        }
        Eigen::VectorXd solved = solver.solve(b);
        if (solver.info() != Eigen::Success) {
            set_error("Eigen SimplicialLDLT solve failed");
            return false;
        }
        for (int row = 0; row < n; ++row) {
            x[static_cast<size_t>(row) * channels + c] = solved(row);
        }
    }
    return true;
#else
    if (!cholesky_decompose(matrix, n)) {
        set_error("Cholesky decomposition failed");
        return false;
    }
    if (!cholesky_solve(matrix, rhs, n, channels, x)) {
        set_error("Cholesky solve failed");
        return false;
    }
    return true;
#endif
}

bool force_dense_solver()
{
    const char* value = std::getenv("VBAKE_FORCE_DENSE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void sparse_multiply(const SparseMatrixRows& matrix, const std::vector<double>& x, std::vector<double>& out)
{
    const int n = static_cast<int>(matrix.rows.size());
    std::fill(out.begin(), out.end(), 0.0);
    for (int row = 0; row < n; ++row) {
        double sum = 0.0;
        for (const auto& entry : matrix.rows[row]) {
            sum += entry.second * x[entry.first];
        }
        out[row] = sum;
    }
}

double dot_vector(const std::vector<double>& a, const std::vector<double>& b)
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

bool solve_pcg_channel(
    const SparseMatrixRows& matrix,
    const std::vector<double>& rhs,
    int n,
    int channels,
    int channel,
    std::vector<double>& x)
{
    std::vector<double> solution(n, 0.0);
    std::vector<double> residual(n, 0.0);
    std::vector<double> z(n, 0.0);
    std::vector<double> direction(n, 0.0);
    std::vector<double> ad(n, 0.0);

    double rhs_norm_sq = 0.0;
    for (int row = 0; row < n; ++row) {
        const double b = rhs[static_cast<size_t>(row) * channels + channel];
        residual[row] = b;
        rhs_norm_sq += b * b;
        const double diag = matrix.diagonal[row];
        if (!std::isfinite(diag) || diag <= 0.0) {
            return false;
        }
        z[row] = b / diag;
        direction[row] = z[row];
    }

    const double absolute_tol = 1e-10;
    const double relative_tol = 1e-8;
    const double threshold = std::max(absolute_tol * absolute_tol, relative_tol * relative_tol * rhs_norm_sq);
    double rz_old = dot_vector(residual, z);
    if (rhs_norm_sq <= absolute_tol * absolute_tol || rz_old <= threshold) {
        for (int row = 0; row < n; ++row) {
            x[static_cast<size_t>(row) * channels + channel] = 0.0;
        }
        return true;
    }

    const int max_iterations = std::max(64, std::min(5000, n * 4));
    for (int iter = 0; iter < max_iterations; ++iter) {
        sparse_multiply(matrix, direction, ad);
        const double denom = dot_vector(direction, ad);
        if (!std::isfinite(denom) || std::abs(denom) <= 1e-30) {
            return false;
        }

        const double alpha = rz_old / denom;
        for (int row = 0; row < n; ++row) {
            solution[row] += alpha * direction[row];
            residual[row] -= alpha * ad[row];
        }

        double residual_norm_sq = dot_vector(residual, residual);
        if (residual_norm_sq <= threshold) {
            for (int row = 0; row < n; ++row) {
                x[static_cast<size_t>(row) * channels + channel] = solution[row];
            }
            return true;
        }

        for (int row = 0; row < n; ++row) {
            z[row] = residual[row] / matrix.diagonal[row];
        }
        const double rz_new = dot_vector(residual, z);
        if (!std::isfinite(rz_new)) {
            return false;
        }
        const double beta = rz_new / rz_old;
        for (int row = 0; row < n; ++row) {
            direction[row] = z[row] + beta * direction[row];
        }
        rz_old = rz_new;
    }

    return false;
}

bool build_sparse_rows(int vertex_count, std::vector<TripletEntry>& triplets, SparseMatrixRows& matrix)
{
    std::sort(triplets.begin(), triplets.end(), [](const TripletEntry& a, const TripletEntry& b) {
        if (a.row != b.row) {
            return a.row < b.row;
        }
        return a.col < b.col;
    });

    matrix.rows.assign(vertex_count, {});
    matrix.diagonal.assign(vertex_count, 0.0);
    int last_row = -1;
    int last_col = -1;
    double sum = 0.0;
    auto flush = [&]() {
        if (last_row < 0 || sum == 0.0) {
            return;
        }
        matrix.rows[last_row].push_back(std::make_pair(last_col, sum));
        if (last_row == last_col) {
            matrix.diagonal[last_row] += sum;
        }
    };

    for (const TripletEntry& entry : triplets) {
        if (entry.row != last_row || entry.col != last_col) {
            flush();
            last_row = entry.row;
            last_col = entry.col;
            sum = entry.value;
        } else {
            sum += entry.value;
        }
    }
    flush();
    return true;
}

bool solve_sparse_least_squares_no_regularization(
    int vertex_count,
    int triangle_count,
    const unsigned int* indices,
    int sample_count,
    const unsigned int* sample_triangles,
    const float* sample_barycentrics,
    const float* sample_values,
    int channels,
    float* out_vertex_values)
{
    const size_t rhs_size = static_cast<size_t>(vertex_count) * channels;
    std::vector<double> rhs(rhs_size, 0.0);
    std::vector<double> lumped(vertex_count, 0.0);
    std::vector<TripletEntry> triplets;
    triplets.reserve(static_cast<size_t>(sample_count) * 9 + static_cast<size_t>(vertex_count));

    for (int s = 0; s < sample_count; ++s) {
        const unsigned int tri_index = sample_triangles[s];
        if (tri_index >= static_cast<unsigned int>(triangle_count)) {
            set_error("sample triangle index out of range");
            return false;
        }

        int tri_vertices[3] = {
            static_cast<int>(indices[static_cast<size_t>(tri_index) * 3 + 0]),
            static_cast<int>(indices[static_cast<size_t>(tri_index) * 3 + 1]),
            static_cast<int>(indices[static_cast<size_t>(tri_index) * 3 + 2]),
        };
        double bary[3] = {
            static_cast<double>(sample_barycentrics[static_cast<size_t>(s) * 3 + 0]),
            static_cast<double>(sample_barycentrics[static_cast<size_t>(s) * 3 + 1]),
            static_cast<double>(sample_barycentrics[static_cast<size_t>(s) * 3 + 2]),
        };

        double bary_sum = bary[0] + bary[1] + bary[2];
        if (!std::isfinite(bary_sum) || std::abs(bary_sum - 1.0) > 1e-3) {
            set_error("sample barycentric coordinates must sum to one");
            return false;
        }

        for (int k = 0; k < 3; ++k) {
            if (tri_vertices[k] < 0 || tri_vertices[k] >= vertex_count) {
                set_error("triangle vertex index out of range");
                return false;
            }
            if (!std::isfinite(bary[k])) {
                set_error("non-finite barycentric coordinate");
                return false;
            }
        }

        for (int i = 0; i < 3; ++i) {
            const int vi = tri_vertices[i];
            const double bi = bary[i];
            lumped[vi] += bi;
            for (int j = 0; j < 3; ++j) {
                triplets.push_back({vi, tri_vertices[j], bi * bary[j]});
            }
            for (int c = 0; c < channels; ++c) {
                const float sample_value = sample_values[static_cast<size_t>(s) * channels + c];
                if (!finite_float(sample_value)) {
                    set_error("non-finite sample value");
                    return false;
                }
                rhs[static_cast<size_t>(vi) * channels + c] += bi * static_cast<double>(sample_value);
            }
        }
    }

    for (int v = 0; v < vertex_count; ++v) {
        triplets.push_back({v, v, lumped[v] <= 0.0 ? 1.0 : 1e-10});
    }

    SparseMatrixRows matrix;
    if (!build_sparse_rows(vertex_count, triplets, matrix)) {
        return false;
    }

    std::vector<double> x(rhs_size, 0.0);
    for (int c = 0; c < channels; ++c) {
        if (!solve_pcg_channel(matrix, rhs, vertex_count, channels, c, x)) {
            set_error("sparse PCG solve failed to converge");
            return false;
        }
    }

    for (size_t i = 0; i < rhs_size; ++i) {
        const double value = x[i];
        if (!std::isfinite(value)) {
            set_error("solver produced non-finite output");
            return false;
        }
        out_vertex_values[i] = static_cast<float>(value);
    }
    return true;
}

} // namespace

extern "C" {

const char* vbake_last_error()
{
    return g_last_error.c_str();
}

int vbake_least_squares(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const unsigned int* indices,
    int sample_count,
    const unsigned int* sample_triangles,
    const float* sample_barycentrics,
    const float* sample_values,
    int channels,
    float regularization_weight,
    float* out_vertex_values)
{
    g_last_error.clear();
    if (!validate_inputs(
            vertex_count,
            triangle_count,
            positions,
            indices,
            sample_count,
            sample_triangles,
            sample_barycentrics,
            sample_values,
            channels,
            out_vertex_values)) {
        return VBAKE_INVALID_ARGUMENT;
    }

    const double reg = std::max(0.0, static_cast<double>(regularization_weight));
    if (reg == 0.0 && !force_dense_solver()) {
        if (solve_sparse_least_squares_no_regularization(
                vertex_count,
                triangle_count,
                indices,
                sample_count,
                sample_triangles,
                sample_barycentrics,
                sample_values,
                channels,
                out_vertex_values)) {
            return VBAKE_SUCCESS;
        }
        g_last_error.clear();
    }

    const size_t matrix_size = static_cast<size_t>(vertex_count) * vertex_count;
    const size_t rhs_size = static_cast<size_t>(vertex_count) * channels;
    std::vector<double> matrix(matrix_size, 0.0);
    std::vector<double> rhs(rhs_size, 0.0);
    std::vector<double> lumped(vertex_count, 0.0);

    for (int s = 0; s < sample_count; ++s) {
        const unsigned int tri_index = sample_triangles[s];
        if (tri_index >= static_cast<unsigned int>(triangle_count)) {
            set_error("sample triangle index out of range");
            return VBAKE_INVALID_ARGUMENT;
        }

        int tri_vertices[3] = {
            static_cast<int>(indices[static_cast<size_t>(tri_index) * 3 + 0]),
            static_cast<int>(indices[static_cast<size_t>(tri_index) * 3 + 1]),
            static_cast<int>(indices[static_cast<size_t>(tri_index) * 3 + 2]),
        };
        double bary[3] = {
            static_cast<double>(sample_barycentrics[static_cast<size_t>(s) * 3 + 0]),
            static_cast<double>(sample_barycentrics[static_cast<size_t>(s) * 3 + 1]),
            static_cast<double>(sample_barycentrics[static_cast<size_t>(s) * 3 + 2]),
        };

        double bary_sum = bary[0] + bary[1] + bary[2];
        if (!std::isfinite(bary_sum) || std::abs(bary_sum - 1.0) > 1e-3) {
            set_error("sample barycentric coordinates must sum to one");
            return VBAKE_INVALID_ARGUMENT;
        }

        for (int k = 0; k < 3; ++k) {
            if (tri_vertices[k] < 0 || tri_vertices[k] >= vertex_count) {
                set_error("triangle vertex index out of range");
                return VBAKE_INVALID_ARGUMENT;
            }
            if (!std::isfinite(bary[k])) {
                set_error("non-finite barycentric coordinate");
                return VBAKE_INVALID_ARGUMENT;
            }
        }

        for (int i = 0; i < 3; ++i) {
            const int vi = tri_vertices[i];
            const double bi = bary[i];
            lumped[vi] += bi;
            for (int j = 0; j < 3; ++j) {
                const int vj = tri_vertices[j];
                matrix[static_cast<size_t>(vi) * vertex_count + vj] += bi * bary[j];
            }
            for (int c = 0; c < channels; ++c) {
                const float sample_value = sample_values[static_cast<size_t>(s) * channels + c];
                if (!finite_float(sample_value)) {
                    set_error("non-finite sample value");
                    return VBAKE_INVALID_ARGUMENT;
                }
                rhs[static_cast<size_t>(vi) * channels + c] += bi * static_cast<double>(sample_value);
            }
        }
    }

    if (reg > 0.0) {
        add_edge_regularization(matrix, vertex_count, triangle_count, positions, indices, reg);
    }

    for (int v = 0; v < vertex_count; ++v) {
        if (lumped[v] <= 0.0) {
            matrix[static_cast<size_t>(v) * vertex_count + v] += 1.0;
        } else {
            matrix[static_cast<size_t>(v) * vertex_count + v] += 1e-10;
        }
    }

    std::vector<double> x(rhs_size, 0.0);
    if (!solve_system(matrix, rhs, vertex_count, channels, x)) {
        return VBAKE_NUMERICAL_FAILURE;
    }

    for (size_t i = 0; i < rhs_size; ++i) {
        const double value = x[i];
        if (!std::isfinite(value)) {
            set_error("solver produced non-finite output");
            return VBAKE_NUMERICAL_FAILURE;
        }
        out_vertex_values[i] = static_cast<float>(value);
    }

    return VBAKE_SUCCESS;
}

}
