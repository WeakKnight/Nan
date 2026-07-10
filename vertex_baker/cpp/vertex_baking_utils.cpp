#include "vertex_baking_utils.h"
#include "tinybvh_visibility.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <string>
#include <thread>
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

bool validate_visibility_inputs(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const float* normals,
    const float* tangents,
    const unsigned int* indices,
    int sample_count,
    const unsigned int* sample_triangles,
    const float* sample_barycentrics,
    const float* sample_raw_cones,
    float* out_encoded_texcoord2)
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
    if (!positions || !normals || !tangents || !indices || !sample_triangles || !sample_barycentrics || !sample_raw_cones
        || !out_encoded_texcoord2) {
        set_error("null visibility input/output pointer");
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
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
    std::vector<double> values;
    std::vector<double> diagonal;
};

struct IncompleteCholesky {
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
    std::vector<double> values;
    std::vector<int> diagonal_indices;
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

double dot(Vec3 a, Vec3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 scale(Vec3 v, double s)
{
    return {v.x * s, v.y * s, v.z * s};
}

Vec3 normalize_or(Vec3 v, Vec3 fallback)
{
    const double len = length(v);
    if (std::isfinite(len) && len > 1e-8) {
        return scale(v, 1.0 / len);
    }
    const double fallback_len = length(fallback);
    if (std::isfinite(fallback_len) && fallback_len > 1e-8) {
        return scale(fallback, 1.0 / fallback_len);
    }
    return {0.0, 0.0, 1.0};
}

Vec3 fallback_tangent(Vec3 normal)
{
    normal = normalize_or(normal, {0.0, 0.0, 1.0});
    Vec3 helper = {0.0, 0.0, 1.0};
    if (std::abs(normal.z) > 0.9) {
        helper = {0.0, 1.0, 0.0};
    }
    return normalize_or(cross(helper, normal), {1.0, 0.0, 0.0});
}

double clamp_double(double value, double lower, double upper)
{
    return std::min(std::max(value, lower), upper);
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
    const int n = static_cast<int>(matrix.diagonal.size());
    std::fill(out.begin(), out.end(), 0.0);
    for (int row = 0; row < n; ++row) {
        double sum = 0.0;
        for (int index = matrix.row_offsets[row]; index < matrix.row_offsets[row + 1]; ++index) {
            sum += matrix.values[index] * x[matrix.column_indices[index]];
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
    std::vector<double>& x,
    double relative_tol = 1e-8,
    double absolute_tol = 1e-10,
    int iteration_multiplier = 4)
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

    const double threshold = std::max(absolute_tol * absolute_tol, relative_tol * relative_tol * rhs_norm_sq);
    double rz_old = dot_vector(residual, z);
    if (rhs_norm_sq <= absolute_tol * absolute_tol || rz_old <= threshold) {
        for (int row = 0; row < n; ++row) {
            x[static_cast<size_t>(row) * channels + channel] = 0.0;
        }
        return true;
    }

    const int max_iterations = std::max(64, std::min(20000, n * std::max(1, iteration_multiplier)));
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

bool solve_pcg_channels_parallel(
    const SparseMatrixRows& matrix,
    const std::vector<double>& rhs,
    int n,
    int channels,
    std::vector<double>& x,
    double relative_tol = 1e-8,
    double absolute_tol = 1e-10,
    int iteration_multiplier = 4)
{
    if (channels <= 0) {
        return false;
    }
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const int worker_count = std::min(
        channels,
        std::max(1, static_cast<int>(hardware_threads == 0 ? 1 : hardware_threads)));
    if (worker_count == 1 || n < 256) {
        for (int channel = 0; channel < channels; ++channel) {
            if (!solve_pcg_channel(
                    matrix,
                    rhs,
                    n,
                    channels,
                    channel,
                    x,
                    relative_tol,
                    absolute_tol,
                    iteration_multiplier)) {
                return false;
            }
        }
        return true;
    }

    std::atomic<int> next_channel(0);
    std::atomic<bool> succeeded(true);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&]() {
            while (succeeded.load(std::memory_order_relaxed)) {
                const int channel = next_channel.fetch_add(1, std::memory_order_relaxed);
                if (channel >= channels) {
                    return;
                }
                if (!solve_pcg_channel(
                        matrix,
                        rhs,
                        n,
                        channels,
                        channel,
                        x,
                        relative_tol,
                        absolute_tol,
                        iteration_multiplier)) {
                    succeeded.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    return succeeded.load(std::memory_order_relaxed);
}

bool build_incomplete_cholesky(const SparseMatrixRows& matrix, IncompleteCholesky& factor)
{
    const int n = static_cast<int>(matrix.diagonal.size());
    factor.row_offsets.assign(static_cast<size_t>(n) + 1, 0);
    factor.column_indices.clear();
    factor.values.clear();
    factor.diagonal_indices.assign(n, -1);
    factor.column_indices.reserve((matrix.column_indices.size() + static_cast<size_t>(n)) / 2);
    factor.values.reserve(factor.column_indices.capacity());

    for (int row = 0; row < n; ++row) {
        factor.row_offsets[row] = static_cast<int>(factor.column_indices.size());
        for (int index = matrix.row_offsets[row]; index < matrix.row_offsets[row + 1]; ++index) {
            const int column = matrix.column_indices[index];
            if (column > row) {
                break;
            }
            factor.column_indices.push_back(column);
            factor.values.push_back(matrix.values[index]);
            if (column == row) {
                factor.diagonal_indices[row] = static_cast<int>(factor.values.size()) - 1;
            }
        }
        factor.row_offsets[row + 1] = static_cast<int>(factor.column_indices.size());
        if (factor.diagonal_indices[row] < 0) {
            return false;
        }
    }

    for (int row = 0; row < n; ++row) {
        const int row_begin = factor.row_offsets[row];
        const int diagonal_index = factor.diagonal_indices[row];
        for (int index = row_begin; index < diagonal_index; ++index) {
            const int column = factor.column_indices[index];
            double value = factor.values[index];
            int left = row_begin;
            int right = factor.row_offsets[column];
            const int right_end = factor.diagonal_indices[column];
            while (left < index && right < right_end) {
                const int left_column = factor.column_indices[left];
                const int right_column = factor.column_indices[right];
                if (left_column == right_column) {
                    value -= factor.values[left] * factor.values[right];
                    ++left;
                    ++right;
                } else if (left_column < right_column) {
                    ++left;
                } else {
                    ++right;
                }
            }
            const double column_diagonal = factor.values[factor.diagonal_indices[column]];
            if (!std::isfinite(column_diagonal) || column_diagonal <= 0.0) {
                return false;
            }
            factor.values[index] = value / column_diagonal;
        }

        double diagonal = matrix.diagonal[row];
        for (int index = row_begin; index < diagonal_index; ++index) {
            diagonal -= factor.values[index] * factor.values[index];
        }
        const double pivot_floor = std::max(std::abs(matrix.diagonal[row]) * 1e-8, 1e-18);
        if (!std::isfinite(diagonal)) {
            return false;
        }
        factor.values[diagonal_index] = std::sqrt(std::max(diagonal, pivot_floor));
    }
    return true;
}

void apply_pcg_preconditioner(
    const SparseMatrixRows& matrix,
    const IncompleteCholesky* incomplete_cholesky,
    const std::vector<double>& input,
    int n,
    int channels,
    const std::array<int, 16>& active_channels,
    int active_count,
    std::vector<double>& output)
{
    if (incomplete_cholesky == nullptr) {
        for (int row = 0; row < n; ++row) {
            const double inverse_diagonal = 1.0 / matrix.diagonal[row];
            const size_t row_offset = static_cast<size_t>(row) * channels;
            for (int active = 0; active < active_count; ++active) {
                const int channel = active_channels[active];
                output[row_offset + channel] = input[row_offset + channel] * inverse_diagonal;
            }
        }
        return;
    }

    const IncompleteCholesky& factor = *incomplete_cholesky;
    for (int row = 0; row < n; ++row) {
        const size_t row_offset = static_cast<size_t>(row) * channels;
        for (int active = 0; active < active_count; ++active) {
            const int channel = active_channels[active];
            double value = input[row_offset + channel];
            for (int index = factor.row_offsets[row]; index < factor.diagonal_indices[row]; ++index) {
                const size_t column_offset = static_cast<size_t>(factor.column_indices[index]) * channels;
                value -= factor.values[index] * output[column_offset + channel];
            }
            output[row_offset + channel] = value / factor.values[factor.diagonal_indices[row]];
        }
    }
    for (int row = n - 1; row >= 0; --row) {
        const size_t row_offset = static_cast<size_t>(row) * channels;
        const double inverse_diagonal = 1.0 / factor.values[factor.diagonal_indices[row]];
        for (int active = 0; active < active_count; ++active) {
            const int channel = active_channels[active];
            output[row_offset + channel] *= inverse_diagonal;
        }
        for (int index = factor.row_offsets[row]; index < factor.diagonal_indices[row]; ++index) {
            const size_t column_offset = static_cast<size_t>(factor.column_indices[index]) * channels;
            const double lower_value = factor.values[index];
            for (int active = 0; active < active_count; ++active) {
                const int channel = active_channels[active];
                output[column_offset + channel] -= lower_value * output[row_offset + channel];
            }
        }
    }
}

bool solve_pcg_multi_rhs(
    const SparseMatrixRows& matrix,
    const std::vector<double>& rhs,
    int n,
    int channels,
    std::vector<double>& x,
    const IncompleteCholesky* incomplete_cholesky,
    double relative_tol = 1e-8,
    double absolute_tol = 1e-10,
    int iteration_multiplier = 4)
{
    if (channels <= 0 || channels > 16) {
        return false;
    }

    const size_t value_count = static_cast<size_t>(n) * channels;
    std::vector<double> solution(value_count, 0.0);
    std::vector<double> residual(value_count, 0.0);
    std::vector<double> preconditioned(value_count, 0.0);
    std::vector<double> direction(value_count, 0.0);
    std::vector<double> ad(value_count, 0.0);
    std::array<double, 16> rhs_norm_sq = {};
    std::array<double, 16> threshold = {};
    std::array<double, 16> rz_old = {};
    std::array<double, 16> denominator = {};
    std::array<double, 16> alpha = {};
    std::array<double, 16> residual_norm_sq = {};
    std::array<double, 16> rz_new = {};
    std::array<double, 16> beta = {};
    std::array<int, 16> active_channels = {};

    for (int row = 0; row < n; ++row) {
        const double diagonal = matrix.diagonal[row];
        if (!std::isfinite(diagonal) || diagonal <= 0.0) {
            return false;
        }
        const size_t row_offset = static_cast<size_t>(row) * channels;
        for (int channel = 0; channel < channels; ++channel) {
            const size_t value_index = row_offset + channel;
            const double b = rhs[value_index];
            residual[value_index] = b;
            rhs_norm_sq[channel] += b * b;
        }
    }

    int active_count = 0;
    for (int channel = 0; channel < channels; ++channel) {
        threshold[channel] = std::max(
            absolute_tol * absolute_tol,
            relative_tol * relative_tol * rhs_norm_sq[channel]);
        if (rhs_norm_sq[channel] > absolute_tol * absolute_tol) {
            active_channels[active_count++] = channel;
        }
    }
    if (active_count == 0) {
        std::fill(x.begin(), x.end(), 0.0);
        return true;
    }

    apply_pcg_preconditioner(
        matrix,
        incomplete_cholesky,
        residual,
        n,
        channels,
        active_channels,
        active_count,
        preconditioned);
    for (int row = 0; row < n; ++row) {
        const size_t row_offset = static_cast<size_t>(row) * channels;
        for (int active = 0; active < active_count; ++active) {
            const int channel = active_channels[active];
            const size_t value_index = row_offset + channel;
            direction[value_index] = preconditioned[value_index];
            rz_old[channel] += residual[value_index] * preconditioned[value_index];
        }
    }
    int initialized_count = 0;
    for (int active = 0; active < active_count; ++active) {
        const int channel = active_channels[active];
        if (rz_old[channel] > threshold[channel]) {
            active_channels[initialized_count++] = channel;
        }
    }
    active_count = initialized_count;
    if (active_count == 0) {
        std::fill(x.begin(), x.end(), 0.0);
        return true;
    }

    const int max_iterations = std::max(64, std::min(20000, n * std::max(1, iteration_multiplier)));
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::fill(denominator.begin(), denominator.end(), 0.0);
        for (int row = 0; row < n; ++row) {
            const size_t row_offset = static_cast<size_t>(row) * channels;
            std::array<double, 16> sums = {};
            for (int index = matrix.row_offsets[row]; index < matrix.row_offsets[row + 1]; ++index) {
                const double matrix_value = matrix.values[index];
                const size_t column_offset = static_cast<size_t>(matrix.column_indices[index]) * channels;
                for (int active = 0; active < active_count; ++active) {
                    const int channel = active_channels[active];
                    sums[channel] += matrix_value * direction[column_offset + channel];
                }
            }
            for (int active = 0; active < active_count; ++active) {
                const int channel = active_channels[active];
                ad[row_offset + channel] = sums[channel];
                denominator[channel] += direction[row_offset + channel] * sums[channel];
            }
        }

        for (int active = 0; active < active_count; ++active) {
            const int channel = active_channels[active];
            if (!std::isfinite(denominator[channel]) || std::abs(denominator[channel]) <= 1e-30) {
                return false;
            }
            alpha[channel] = rz_old[channel] / denominator[channel];
            residual_norm_sq[channel] = 0.0;
        }
        for (int row = 0; row < n; ++row) {
            const size_t row_offset = static_cast<size_t>(row) * channels;
            for (int active = 0; active < active_count; ++active) {
                const int channel = active_channels[active];
                const size_t value_index = row_offset + channel;
                solution[value_index] += alpha[channel] * direction[value_index];
                residual[value_index] -= alpha[channel] * ad[value_index];
                residual_norm_sq[channel] += residual[value_index] * residual[value_index];
            }
        }

        int remaining_count = 0;
        for (int active = 0; active < active_count; ++active) {
            const int channel = active_channels[active];
            if (residual_norm_sq[channel] > threshold[channel]) {
                active_channels[remaining_count++] = channel;
            }
        }
        active_count = remaining_count;
        if (active_count == 0) {
            x = std::move(solution);
            return true;
        }

        apply_pcg_preconditioner(
            matrix,
            incomplete_cholesky,
            residual,
            n,
            channels,
            active_channels,
            active_count,
            preconditioned);
        std::fill(rz_new.begin(), rz_new.end(), 0.0);
        for (int row = 0; row < n; ++row) {
            const size_t row_offset = static_cast<size_t>(row) * channels;
            for (int active = 0; active < active_count; ++active) {
                const int channel = active_channels[active];
                const size_t value_index = row_offset + channel;
                rz_new[channel] += residual[value_index] * preconditioned[value_index];
            }
        }
        for (int active = 0; active < active_count; ++active) {
            const int channel = active_channels[active];
            if (!std::isfinite(rz_new[channel])) {
                return false;
            }
            beta[channel] = rz_new[channel] / rz_old[channel];
        }
        for (int row = 0; row < n; ++row) {
            const size_t row_offset = static_cast<size_t>(row) * channels;
            for (int active = 0; active < active_count; ++active) {
                const int channel = active_channels[active];
                const size_t value_index = row_offset + channel;
                direction[value_index] = preconditioned[value_index] + beta[channel] * direction[value_index];
            }
        }
        for (int active = 0; active < active_count; ++active) {
            const int channel = active_channels[active];
            rz_old[channel] = rz_new[channel];
        }
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

    matrix.row_offsets.assign(static_cast<size_t>(vertex_count) + 1, 0);
    matrix.column_indices.clear();
    matrix.values.clear();
    matrix.column_indices.reserve(triplets.size());
    matrix.values.reserve(triplets.size());
    matrix.diagonal.assign(vertex_count, 0.0);
    int last_row = -1;
    int last_col = -1;
    double sum = 0.0;
    auto flush = [&]() {
        if (last_row < 0 || sum == 0.0) {
            return;
        }
        matrix.column_indices.push_back(last_col);
        matrix.values.push_back(sum);
        matrix.row_offsets[static_cast<size_t>(last_row) + 1] += 1;
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
    for (int row = 0; row < vertex_count; ++row) {
        matrix.row_offsets[row + 1] += matrix.row_offsets[row];
    }
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
    if (!solve_pcg_channels_parallel(matrix, rhs, vertex_count, channels, x)) {
        set_error("sparse parallel PCG solve failed to converge");
        return false;
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

Vec3 normalize_zero(Vec3 value)
{
    const double len = length(value);
    if (!std::isfinite(len) || len <= 1e-20) {
        return {0.0, 0.0, 0.0};
    }
    return scale(value, 1.0 / len);
}

void pmr_sh_basis(Vec3 direction, double out[16])
{
    const double x = direction.x;
    const double y = direction.y;
    const double z = direction.z;
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;
    out[0] = 0.28209479177387814;
    out[1] = -0.4886025119029199 * y;
    out[2] = 0.4886025119029199 * z;
    out[3] = -0.4886025119029199 * x;
    out[4] = 1.0925484305920792 * x * y;
    out[5] = -1.0925484305920792 * y * z;
    out[6] = 0.31539156525252005 * (-1.0 + 3.0 * z2);
    out[7] = -1.0925484305920792 * x * z;
    out[8] = 0.5462742152960396 * (x2 - y2);
    out[9] = -0.5900435899266435 * (3.0 * x2 * y - y2 * y);
    out[10] = 2.890611442640554 * x * y * z;
    out[11] = -0.4570457994644658 * y * (-1.0 + 5.0 * z2);
    out[12] = 0.3731763325901154 * z * (-3.0 + 5.0 * z2);
    out[13] = -0.4570457994644658 * x * (-1.0 + 5.0 * z2);
    out[14] = 1.445305721320277 * (x2 - y2) * z;
    out[15] = -0.5900435899266435 * (x * x2 - 3.0 * x * y2);
}

void pmr_a_hat_transpose(const float* input, double output[16])
{
    std::fill(output, output + 16, 0.0);
    output[4] = 1.58533 * input[4] + 0.457646 * input[5] + 1.58533 * input[6] - 1.37294 * input[7] - 0.915291 * input[8];
    output[5] = 2.11378 * input[4];
    output[6] = 1.05689 * input[4] + 1.83058 * input[5] - 1.83058 * input[7] - 1.83058 * input[8];
    output[7] = -2.28823 * input[7];
    output[8] = -2.28823 * input[5];
    output[9] = 1.498 * input[10] - 1.33985 * input[12] + 0.864869 * input[14] + 2.11849 * input[15];
    output[10] = -2.52644 * input[13];
    output[11] = 2.18796 * input[11] - 1.26322 * input[13];
    output[12] = 2.36854 * input[15];
    output[13] = -1.18427 * input[9] + 1.67481 * input[10] + 1.52889 * input[11] - 2.64811 * input[13] + 0.966953 * input[14] + 1.18427 * input[15];
    output[14] = 2.23308 * input[10];
    output[15] = 1.18427 * input[9] - 0.55827 * input[10] - 1.52889 * input[11] + 2.64811 * input[13] + 0.966953 * input[14] + 1.18427 * input[15];
}

double pmr_cone_coefficient(double aperture, int coefficient)
{
    constexpr double kPi = 3.14159265358979323846;
    switch (coefficient) {
    case 0:
        return -std::sqrt(kPi) * (-1.0 + std::cos(aperture));
    case 2:
        return 0.5 * std::sqrt(3.0 * kPi) * std::sin(aperture) * std::sin(aperture);
    case 6:
        return 0.5 * std::sqrt(5.0 * kPi) * std::sin(aperture) * std::sin(aperture) * std::cos(aperture);
    case 12:
        return std::sqrt(7.0 * kPi) / 16.0 * std::sin(aperture) * std::sin(aperture)
            * (5.0 * std::cos(2.0 * aperture) + 3.0);
    default:
        return 0.0;
    }
}

double pmr_cone_derivative(double aperture, int coefficient)
{
    constexpr double kPi = 3.14159265358979323846;
    switch (coefficient) {
    case 0:
        return std::sqrt(kPi) * std::sin(aperture);
    case 2:
        return std::sqrt(3.0 * kPi) * std::cos(aperture) * std::sin(aperture);
    case 6:
        return (3.0 * std::sqrt(5.0 * kPi) * std::sin(3.0 * aperture)
            - std::sqrt(5.0 * kPi) * std::sin(aperture)) / 8.0;
    case 12:
        return (5.0 * std::sqrt(7.0 * kPi) * std::sin(4.0 * aperture)
            - 2.0 * std::sqrt(7.0 * kPi) * std::sin(2.0 * aperture)) / 16.0;
    default:
        return 0.0;
    }
}

double pmr_aperture_equation(const double sh[16], double aperture)
{
    double numerator = 0.0;
    double denominator = 0.0;
    double lhs = 0.0;
    double rhs = 0.0;
    for (int coefficient = 0; coefficient < 16; ++coefficient) {
        const double ca = pmr_cone_coefficient(aperture, coefficient);
        const double dca = pmr_cone_derivative(aperture, coefficient);
        lhs += sh[coefficient] * dca;
        numerator += sh[coefficient] * ca;
        denominator += ca * ca;
        rhs += ca * dca;
    }
    return lhs - rhs * numerator / denominator;
}

void solve_pmr_cone(const double sh[16], double& aperture, double& cone_scale)
{
    constexpr double kPi = 3.14159265358979323846;
    constexpr int kSplitCount = 20;
    double x1 = 0.001;
    double x2 = kPi - 0.001;
    const double step = (x2 - x1) / kSplitCount;
    double values[kSplitCount + 1];
    for (int i = 0; i <= kSplitCount; ++i) {
        values[i] = pmr_aperture_equation(sh, x1 + step * i);
    }

    bool found = false;
    for (int i = 0; i < kSplitCount && !found; ++i) {
        for (int j = i + 1; j <= kSplitCount; ++j) {
            if (values[i] * values[j] < 0.0) {
                x2 = x1 + step * j;
                x1 = x1 + step * i;
                found = true;
                break;
            }
        }
    }
    if (!found) {
        aperture = sh[0] > 0.1 ? kPi - 0.001 : 0.001;
        cone_scale = 1.0;
        return;
    }

    double dx;
    if (pmr_aperture_equation(sh, x1) < 0.0) {
        dx = x2 - x1;
        aperture = x1;
    } else {
        dx = x1 - x2;
        aperture = x2;
    }
    for (int i = 0; i < 10; ++i) {
        dx *= 0.5;
        const double midpoint = aperture + dx;
        const double value = pmr_aperture_equation(sh, midpoint);
        if (value <= 0.0) {
            aperture = midpoint;
        }
        if (value == 0.0) {
            break;
        }
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (int coefficient = 0; coefficient < 16; ++coefficient) {
        const double ca = pmr_cone_coefficient(aperture, coefficient);
        numerator += sh[coefficient] * ca;
        denominator += ca * ca;
    }
    cone_scale = numerator / denominator;
}

Vec3 inverse_rotate(Vec3 new_x, Vec3 new_z, Vec3 new_y, Vec3 value)
{
    const Vec3 cross_zy = cross(new_z, new_y);
    const double determinant = dot(new_x, cross_zy);
    if (!std::isfinite(determinant) || std::abs(determinant) <= 1e-20) {
        return {0.0, 0.0, 0.0};
    }
    const double inv_determinant = 1.0 / determinant;
    return {
        dot(value, cross_zy) * inv_determinant,
        dot(value, cross(new_y, new_x)) * inv_determinant,
        dot(value, cross(new_x, new_z)) * inv_determinant,
    };
}

void rotate_pmr_sh_to_zonal(const float* sh, Vec3& axis, double zonal[16])
{
    const Vec3 linear = {
        -static_cast<double>(sh[3]),
        -static_cast<double>(sh[1]),
        static_cast<double>(sh[2]),
    };
    axis = normalize_zero(linear);

    Vec3 new_x;
    Vec3 new_z;
    if (axis.y >= 0.999) {
        new_x = {1.0, 0.0, 0.0};
        new_z = {0.0, 0.0, 1.0};
    } else if (axis.y <= -0.999) {
        new_x = {1.0, 0.0, 0.0};
        new_z = {0.0, 0.0, -1.0};
    } else {
        new_x = normalize_zero(cross({0.0, 1.0, 0.0}, axis));
        new_z = normalize_zero(cross(new_x, axis));
    }

    static constexpr double kLobes[7][2] = {
        {3.1416, 2.6180},
        {1.5708, -2.6180},
        {1.5708, 1.5708},
        {2.0344, -3.1416},
        {2.0344, -1.5708},
        {2.0344, -0.5236},
        {2.0344, 1.5708},
    };
    double z[16];
    pmr_a_hat_transpose(sh, z);
    std::fill(zonal, zonal + 16, 0.0);
    zonal[0] = sh[0];
    zonal[2] = inverse_rotate(new_x, new_z, axis, linear).z;

    for (int lobe = 0; lobe < 7; ++lobe) {
        const double theta = kLobes[lobe][0];
        const double phi = kLobes[lobe][1];
        const double sin_theta = std::sin(theta);
        const Vec3 direction = {
            sin_theta * std::cos(phi),
            sin_theta * std::sin(phi),
            std::cos(theta),
        };
        const Vec3 rotated = inverse_rotate(new_x, new_z, axis, direction);
        double basis[16];
        pmr_sh_basis(rotated, basis);
        if (lobe < 5) {
            zonal[6] += basis[6] * z[4 + lobe];
        }
        zonal[12] += basis[12] * z[9 + lobe];
    }
}

bool build_pmr_sparse_system(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const unsigned int* indices,
    const float* triangle_areas,
    double edge_regularization,
    SparseMatrixRows& matrix)
{
    std::vector<TripletEntry> triplets;
    triplets.reserve(static_cast<size_t>(triangle_count) * 25);
    std::vector<std::array<Vec3, 3>> gradients(static_cast<size_t>(triangle_count));
    std::map<std::pair<int, int>, std::vector<int>> edge_links;

    for (int triangle = 0; triangle < triangle_count; ++triangle) {
        int ids[3];
        for (int corner = 0; corner < 3; ++corner) {
            const unsigned int id = indices[static_cast<size_t>(triangle) * 3 + corner];
            if (id >= static_cast<unsigned int>(vertex_count)) {
                set_error("triangle vertex index out of range");
                return false;
            }
            ids[corner] = static_cast<int>(id);
        }
        const double area = triangle_areas[triangle];
        if (!std::isfinite(area) || area <= 0.0) {
            set_error("triangle areas must be finite and positive");
            return false;
        }
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                triplets.push_back({ids[row], ids[col], area * (row == col ? 1.0 / 6.0 : 1.0 / 12.0)});
            }
            const int a = std::min(ids[row], ids[(row + 1) % 3]);
            const int b = std::max(ids[row], ids[(row + 1) % 3]);
            edge_links[std::make_pair(a, b)].push_back(triangle);
        }

        const Vec3 v0 = load_position(positions, ids[0]);
        const Vec3 v1 = load_position(positions, ids[1]);
        const Vec3 v2 = load_position(positions, ids[2]);
        const double gradient_scale = 0.5 / std::sqrt(area);
        gradients[triangle][0] = scale(cross(normalize_zero(cross(sub(v1, v0), sub(v2, v1))), sub(v2, v1)), gradient_scale);
        gradients[triangle][1] = scale(cross(normalize_zero(cross(sub(v2, v1), sub(v0, v2))), sub(v0, v2)), gradient_scale);
        gradients[triangle][2] = scale(cross(normalize_zero(cross(sub(v0, v2), sub(v1, v0))), sub(v1, v0)), gradient_scale);
    }

    if (edge_regularization > 0.0) {
        for (const auto& link : edge_links) {
            const std::vector<int>& linked = link.second;
            for (size_t a = 0; a + 1 < linked.size(); ++a) {
                const int triangle_a = linked[a];
                int ids[4] = {
                    static_cast<int>(indices[static_cast<size_t>(triangle_a) * 3 + 0]),
                    static_cast<int>(indices[static_cast<size_t>(triangle_a) * 3 + 1]),
                    static_cast<int>(indices[static_cast<size_t>(triangle_a) * 3 + 2]),
                    0,
                };
                for (size_t b = a + 1; b < linked.size(); ++b) {
                    const int triangle_b = linked[b];
                    Vec3 redge[4] = {
                        gradients[triangle_a][0],
                        gradients[triangle_a][1],
                        gradients[triangle_a][2],
                        {0.0, 0.0, 0.0},
                    };
                    for (int corner_b = 0; corner_b < 3; ++corner_b) {
                        const int vertex_b = static_cast<int>(indices[static_cast<size_t>(triangle_b) * 3 + corner_b]);
                        int target = 3;
                        for (int corner_a = 0; corner_a < 3; ++corner_a) {
                            if (ids[corner_a] == vertex_b) {
                                target = corner_a;
                                break;
                            }
                        }
                        redge[target] = sub(redge[target], gradients[triangle_b][corner_b]);
                        if (target == 3) {
                            ids[3] = vertex_b;
                        }
                    }
                    const double area = static_cast<double>(triangle_areas[triangle_a]) + triangle_areas[triangle_b];
                    for (int row = 0; row < 4; ++row) {
                        for (int col = 0; col < 4; ++col) {
                            triplets.push_back({ids[row], ids[col], edge_regularization * area * dot(redge[row], redge[col])});
                        }
                    }
                }
            }
        }
    }

    return build_sparse_rows(vertex_count, triplets, matrix);
}

bool build_pmr_rhs(
    int vertex_count,
    int triangle_count,
    const unsigned int* indices,
    const float* triangle_areas,
    int samples_per_triangle,
    const float* sample_barycentrics,
    const float* sample_sh16,
    std::vector<double>& rhs)
{
    constexpr int kChannels = 16;
    const size_t rhs_size = static_cast<size_t>(vertex_count) * kChannels;
    const size_t sample_count = static_cast<size_t>(triangle_count) * samples_per_triangle;
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    int worker_count = 1;
    if (sample_count >= 65536) {
        constexpr size_t kMaxPartialRhsBytes = 256ull * 1024ull * 1024ull;
        const size_t bytes_per_worker = std::max<size_t>(rhs_size * sizeof(double), 1);
        const size_t memory_limited_workers = std::max<size_t>(1, kMaxPartialRhsBytes / bytes_per_worker);
        worker_count = std::min(
            triangle_count,
            std::min(
                32,
                std::min(
                    std::max(1, static_cast<int>(hardware_threads == 0 ? 1 : hardware_threads)),
                    static_cast<int>(std::min<size_t>(
                        memory_limited_workers,
                        static_cast<size_t>(std::numeric_limits<int>::max()))))));
    }

    std::vector<double> partial_rhs(static_cast<size_t>(worker_count) * rhs_size, 0.0);
    std::atomic<int> error_code(0);
    auto accumulate_range = [&](int worker, int triangle_begin, int triangle_end) {
        double* local_rhs = partial_rhs.data() + static_cast<size_t>(worker) * rhs_size;
        for (int triangle = triangle_begin; triangle < triangle_end; ++triangle) {
            if (error_code.load(std::memory_order_relaxed) != 0) {
                return;
            }
            const double sample_weight = static_cast<double>(triangle_areas[triangle]) / samples_per_triangle;
            const int ids[3] = {
                static_cast<int>(indices[static_cast<size_t>(triangle) * 3 + 0]),
                static_cast<int>(indices[static_cast<size_t>(triangle) * 3 + 1]),
                static_cast<int>(indices[static_cast<size_t>(triangle) * 3 + 2]),
            };
            double* rhs0 = local_rhs + static_cast<size_t>(ids[0]) * kChannels;
            double* rhs1 = local_rhs + static_cast<size_t>(ids[1]) * kChannels;
            double* rhs2 = local_rhs + static_cast<size_t>(ids[2]) * kChannels;
            for (int sample_in_triangle = 0; sample_in_triangle < samples_per_triangle; ++sample_in_triangle) {
                const size_t sample = static_cast<size_t>(triangle) * samples_per_triangle + sample_in_triangle;
                const float* bary = sample_barycentrics + sample * 3;
                if (!finite_float(bary[0]) || !finite_float(bary[1]) || !finite_float(bary[2])) {
                    error_code.store(1, std::memory_order_relaxed);
                    return;
                }
                const double bary_sum = static_cast<double>(bary[0]) + bary[1] + bary[2];
                if (!std::isfinite(bary_sum) || std::abs(bary_sum - 1.0) > 1e-3) {
                    error_code.store(2, std::memory_order_relaxed);
                    return;
                }
                const float* sample_sh = sample_sh16 + sample * kChannels;
                for (int channel = 0; channel < kChannels; ++channel) {
                    const float sample_value = sample_sh[channel];
                    if (!finite_float(sample_value)) {
                        error_code.store(3, std::memory_order_relaxed);
                        return;
                    }
                    const double weighted_value = sample_weight * static_cast<double>(sample_value);
                    rhs0[channel] += static_cast<double>(bary[0]) * weighted_value;
                    rhs1[channel] += static_cast<double>(bary[1]) * weighted_value;
                    rhs2[channel] += static_cast<double>(bary[2]) * weighted_value;
                }
            }
        }
    };

    if (worker_count == 1) {
        accumulate_range(0, 0, triangle_count);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(worker_count - 1);
        for (int worker = 1; worker < worker_count; ++worker) {
            const int begin = triangle_count * worker / worker_count;
            const int end = triangle_count * (worker + 1) / worker_count;
            workers.emplace_back(accumulate_range, worker, begin, end);
        }
        accumulate_range(0, 0, triangle_count / worker_count);
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

    switch (error_code.load(std::memory_order_relaxed)) {
    case 1:
        set_error("non-finite PMR sample barycentric coordinate");
        return false;
    case 2:
        set_error("PMR sample barycentric coordinates must sum to one");
        return false;
    case 3:
        set_error("non-finite PMR sample SH coefficient");
        return false;
    default:
        break;
    }

    rhs.assign(partial_rhs.begin(), partial_rhs.begin() + static_cast<std::ptrdiff_t>(rhs_size));
    for (int worker = 1; worker < worker_count; ++worker) {
        const double* local_rhs = partial_rhs.data() + static_cast<size_t>(worker) * rhs_size;
        for (size_t index = 0; index < rhs_size; ++index) {
            rhs[index] += local_rhs[index];
        }
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

int vbake_visibility_least_squares(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const float* normals,
    const float* tangents,
    const unsigned int* indices,
    int sample_count,
    const unsigned int* sample_triangles,
    const float* sample_barycentrics,
    const float* sample_raw_cones,
    float regularization_weight,
    float* out_vertex_cones,
    float* out_encoded_texcoord2)
{
    g_last_error.clear();
    if (!validate_visibility_inputs(
            vertex_count,
            triangle_count,
            positions,
            normals,
            tangents,
            indices,
            sample_count,
            sample_triangles,
            sample_barycentrics,
            sample_raw_cones,
            out_encoded_texcoord2)) {
        return VBAKE_INVALID_ARGUMENT;
    }

    constexpr int kConeChannels = 5;
    constexpr double kHalfPi = 1.57079632679489661923;
    std::vector<float> solved(static_cast<size_t>(vertex_count) * kConeChannels, 0.0f);
    const int status = vbake_least_squares(
        vertex_count,
        triangle_count,
        positions,
        indices,
        sample_count,
        sample_triangles,
        sample_barycentrics,
        sample_raw_cones,
        kConeChannels,
        regularization_weight,
        solved.data());
    if (status != VBAKE_SUCCESS) {
        return status;
    }

    std::vector<unsigned char> constrained(static_cast<size_t>(vertex_count), 0);
    for (int s = 0; s < sample_count; ++s) {
        const unsigned int tri_index = sample_triangles[s];
        if (tri_index >= static_cast<unsigned int>(triangle_count)) {
            set_error("sample triangle index out of range");
            return VBAKE_INVALID_ARGUMENT;
        }
        for (int k = 0; k < 3; ++k) {
            const unsigned int vertex_index = indices[static_cast<size_t>(tri_index) * 3 + k];
            if (vertex_index >= static_cast<unsigned int>(vertex_count)) {
                set_error("triangle vertex index out of range");
                return VBAKE_INVALID_ARGUMENT;
            }
            constrained[vertex_index] = 1;
        }
    }

    for (int v = 0; v < vertex_count; ++v) {
        const size_t vertex_offset3 = static_cast<size_t>(v) * 3;
        const size_t vertex_offset4 = static_cast<size_t>(v) * 4;
        const size_t cone_offset = static_cast<size_t>(v) * kConeChannels;

        Vec3 normal = {
            static_cast<double>(normals[vertex_offset3 + 0]),
            static_cast<double>(normals[vertex_offset3 + 1]),
            static_cast<double>(normals[vertex_offset3 + 2]),
        };
        Vec3 fallback = fallback_tangent(normal);
        normal = normalize_or(normal, {0.0, 0.0, 1.0});

        Vec3 tangent = {
            static_cast<double>(tangents[vertex_offset4 + 0]),
            static_cast<double>(tangents[vertex_offset4 + 1]),
            static_cast<double>(tangents[vertex_offset4 + 2]),
        };
        tangent = normalize_or(tangent, fallback);
        const double tangent_w = tangents[vertex_offset4 + 3] < 0.0f ? -1.0 : 1.0;
        Vec3 bitangent = normalize_or(scale(cross(normal, tangent), tangent_w), fallback);
        Vec3 ortho_tangent = normalize_or(cross(bitangent, normal), tangent);

        Vec3 direction = {
            static_cast<double>(solved[cone_offset + 0]),
            static_cast<double>(solved[cone_offset + 1]),
            static_cast<double>(solved[cone_offset + 2]),
        };
        direction = normalize_or(direction, normal);
        double aperture = clamp_double(static_cast<double>(solved[cone_offset + 3]), 0.0, kHalfPi);
        double cone_scale = clamp_double(static_cast<double>(solved[cone_offset + 4]), 0.0, 1.0);
        if (!constrained[v]) {
            direction = normal;
            aperture = kHalfPi;
            cone_scale = 1.0;
        }

        if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z)
            || !std::isfinite(aperture) || !std::isfinite(cone_scale)) {
            set_error("visibility encode produced non-finite output");
            return VBAKE_NUMERICAL_FAILURE;
        }

        if (out_vertex_cones) {
            out_vertex_cones[cone_offset + 0] = static_cast<float>(direction.x);
            out_vertex_cones[cone_offset + 1] = static_cast<float>(direction.y);
            out_vertex_cones[cone_offset + 2] = static_cast<float>(direction.z);
            out_vertex_cones[cone_offset + 3] = static_cast<float>(aperture);
            out_vertex_cones[cone_offset + 4] = static_cast<float>(cone_scale);
        }

        out_encoded_texcoord2[vertex_offset4 + 0] = static_cast<float>(clamp_double(dot(direction, bitangent), -1.0, 1.0));
        out_encoded_texcoord2[vertex_offset4 + 1] = static_cast<float>(std::atan2(dot(direction, ortho_tangent), dot(direction, normal)));
        out_encoded_texcoord2[vertex_offset4 + 2] = static_cast<float>(aperture);
        out_encoded_texcoord2[vertex_offset4 + 3] = static_cast<float>(cone_scale);
    }

    return VBAKE_SUCCESS;
}

int vbake_pmr_visibility_sh_least_squares(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const unsigned int* indices,
    const float* triangle_areas,
    int samples_per_triangle,
    const float* sample_barycentrics,
    const float* sample_sh16,
    float edge_regularization,
    float* out_vertex_sh16)
{
    g_last_error.clear();
    if (vertex_count <= 0 || triangle_count <= 0 || samples_per_triangle <= 0) {
        set_error("PMR vertex, triangle, and per-triangle sample counts must be positive");
        return VBAKE_INVALID_ARGUMENT;
    }
    if (!positions || !indices || !triangle_areas || !sample_barycentrics || !sample_sh16 || !out_vertex_sh16) {
        set_error("null PMR visibility input/output pointer");
        return VBAKE_INVALID_ARGUMENT;
    }

    SparseMatrixRows matrix;
    if (!build_pmr_sparse_system(
            vertex_count,
            triangle_count,
            positions,
            indices,
            triangle_areas,
            std::max(0.0, static_cast<double>(edge_regularization)),
            matrix)) {
        return VBAKE_INVALID_ARGUMENT;
    }

    constexpr int kChannels = 16;
    const size_t rhs_size = static_cast<size_t>(vertex_count) * kChannels;
    std::vector<double> rhs;
    if (!build_pmr_rhs(
            vertex_count,
            triangle_count,
            indices,
            triangle_areas,
            samples_per_triangle,
            sample_barycentrics,
            sample_sh16,
            rhs)) {
        return VBAKE_INVALID_ARGUMENT;
    }

    std::vector<double> solved(rhs_size, 0.0);
    bool solved_ok = solve_pcg_channels_parallel(
            matrix,
            rhs,
            vertex_count,
            kChannels,
            solved,
            1e-11,
            1e-13,
            20);
    if (!solved_ok) {
        IncompleteCholesky incomplete_cholesky;
        const IncompleteCholesky* preconditioner =
            build_incomplete_cholesky(matrix, incomplete_cholesky) ? &incomplete_cholesky : nullptr;
        std::fill(solved.begin(), solved.end(), 0.0);
        solved_ok = solve_pcg_multi_rhs(
            matrix,
            rhs,
            vertex_count,
            kChannels,
            solved,
            preconditioner,
            1e-11,
            1e-13,
            20);
    }
    if (!solved_ok) {
        set_error("PMR sparse PCG solve failed to converge");
        return VBAKE_NUMERICAL_FAILURE;
    }
    for (size_t i = 0; i < rhs_size; ++i) {
        if (!std::isfinite(solved[i])) {
            set_error("PMR solver produced non-finite output");
            return VBAKE_NUMERICAL_FAILURE;
        }
        out_vertex_sh16[i] = static_cast<float>(solved[i]);
    }
    return VBAKE_SUCCESS;
}

int vbake_pmr_sh_to_cones(
    int vertex_count,
    const float* vertex_sh16,
    const float* fallback_normals,
    float* out_vertex_cones)
{
    g_last_error.clear();
    if (vertex_count <= 0 || !vertex_sh16 || !fallback_normals || !out_vertex_cones) {
        set_error("invalid PMR SH-to-cone input/output");
        return VBAKE_INVALID_ARGUMENT;
    }
    for (int vertex = 0; vertex < vertex_count; ++vertex) {
        const float* sh = vertex_sh16 + static_cast<size_t>(vertex) * 16;
        for (int coefficient = 0; coefficient < 16; ++coefficient) {
            if (!finite_float(sh[coefficient])) {
                set_error("non-finite PMR vertex SH coefficient");
                return VBAKE_INVALID_ARGUMENT;
            }
        }
        Vec3 axis;
        double zonal[16];
        rotate_pmr_sh_to_zonal(sh, axis, zonal);
        double aperture;
        double cone_scale;
        solve_pmr_cone(zonal, aperture, cone_scale);
        if (!std::isfinite(axis.x) || !std::isfinite(axis.y) || !std::isfinite(axis.z)
            || !std::isfinite(aperture) || !std::isfinite(cone_scale)) {
            set_error("PMR SH-to-cone produced non-finite output");
            return VBAKE_NUMERICAL_FAILURE;
        }
        float* cone = out_vertex_cones + static_cast<size_t>(vertex) * 5;
        cone[0] = static_cast<float>(axis.x);
        cone[1] = static_cast<float>(axis.y);
        cone[2] = static_cast<float>(axis.z);
        cone[3] = static_cast<float>(aperture);
        cone[4] = static_cast<float>(cone_scale);
    }
    return VBAKE_SUCCESS;
}

int vbake_tinybvh_has_bvh8_cpu()
{
    return tinybvh_has_bvh8_cpu() ? 1 : 0;
}

int vbake_pmr_visibility_sh_tinybvh(
    int vertex_count,
    int triangle_count,
    const float* positions,
    const unsigned int* indices,
    int sample_count,
    const float* sample_positions,
    const float* sample_normals,
    int ray_count,
    float max_distance,
    float self_bias,
    int thread_count,
    int layout,
    float* out_sample_sh16,
    double* out_build_milliseconds,
    double* out_trace_milliseconds,
    unsigned long long* out_visible_ray_count,
    int* out_thread_count)
{
    g_last_error.clear();
    TinyBvhTraceTimings timings;
    std::string error;
    if (!trace_pmr_visibility_sh_tinybvh(
            vertex_count,
            triangle_count,
            positions,
            indices,
            sample_count,
            sample_positions,
            sample_normals,
            ray_count,
            max_distance,
            self_bias,
            thread_count,
            layout,
            out_sample_sh16,
            timings,
            error)) {
        set_error(error);
        return VBAKE_INVALID_ARGUMENT;
    }
    if (out_build_milliseconds) {
        *out_build_milliseconds = timings.build_milliseconds;
    }
    if (out_trace_milliseconds) {
        *out_trace_milliseconds = timings.trace_milliseconds;
    }
    if (out_visible_ray_count) {
        *out_visible_ray_count = static_cast<unsigned long long>(timings.visible_ray_count);
    }
    if (out_thread_count) {
        *out_thread_count = timings.thread_count;
    }
    return VBAKE_SUCCESS;
}

}
