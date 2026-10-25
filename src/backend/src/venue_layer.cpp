/**
 * @file venue_layer.cpp
 * @brief 场外层的体素累积与冻结。
 */

#include "backend/venue_layer.hpp"

#include <cmath>

#include "backend/geometry.hpp"

namespace backend {
namespace {
/// 体素坐标打包成一个键：每轴 21 位，覆盖 ±10^5 个体素。
std::int64_t voxelKey(const Eigen::Vector3d& point, double voxel) {
    const auto axis = [&](double v) {
        return (static_cast<std::int64_t>(std::floor(v / voxel)) + (1LL << 20)) & ((1LL << 21) - 1);
    };
    return (axis(point.x()) << 42) | (axis(point.y()) << 21) | axis(point.z());
}
}  // namespace

bool VenueLayer::insert(const Cloud& points_map, double timestamp, std::uint64_t frame_id) {
    if (frozen_ || config_.voxel <= 0.0)
        return false;
    if (first_timestamp_ < 0.0)
        first_timestamp_ = timestamp;
    ++frames_;
    for (const auto& point : points_map.points) {
        const Eigen::Vector3d p = point.getVector3fMap().cast<double>();
        if (!p.allFinite())
            continue;
        Cell& cell = cells_[voxelKey(p, config_.voxel)];
        cell.sum += p;
        ++cell.points;
        if (cell.frames == 0 || cell.last_frame != frame_id) {
            ++cell.frames;
            cell.last_frame = frame_id;
        }
    }
    if (config_.build_seconds > 0.0 && timestamp - first_timestamp_ >= config_.build_seconds) {
        freeze();
        return true;
    }
    return false;
}

void VenueLayer::freeze() {
    cloud_.reset(new Cloud);
    for (const auto& [key, cell] : cells_) {
        if (cell.frames < config_.min_frames || cell.points <= 0)
            continue;
        const Eigen::Vector3d centroid = cell.sum / static_cast<double>(cell.points);
        PointType point;
        point.getVector3fMap() = centroid.cast<float>();
        cloud_->push_back(point);
    }
    cloud_->width = static_cast<std::uint32_t>(cloud_->size());
    cloud_->height = 1;
    cloud_->is_dense = true;
    frozen_ = true;
    ++revision_;
}
Cloud::ConstPtr VenueTargets::merged(int index, std::uint64_t revision, const Cloud& target,
                                     const Cloud& venue_map, const Eigen::Matrix4d& reference_pose,
                                     double radius) {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto key = std::make_pair(index, revision);
    const auto found = merged_.find(key);
    if (found != merged_.end())
        return found->second;
    if (merged_.size() >= capacity_)
        merged_.clear();
    const Eigen::Vector3d centre = reference_pose.block<3, 1>(0, 3);
    Cloud near;
    for (const auto& point : venue_map.points) {
        if (radius <= 0.0 || (point.getVector3fMap().cast<double>() - centre).norm() <= radius)
            near.push_back(point);
    }
    Cloud::Ptr out(new Cloud(target));
    *out += *transformCloud(near, reference_pose.inverse());
    out->width = static_cast<std::uint32_t>(out->size());
    out->height = 1;
    merged_.emplace(key, out);
    return out;
}
}  // namespace backend
