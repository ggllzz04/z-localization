/**
 * @file venue_layer.hpp
 * @brief 比赛中跟踪配准用的场外层：场地围挡以上的静止点，开局定稳后在线累积。
 */

#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "backend/config.hpp"
#include "backend/types.hpp"

namespace backend {
/// map 系体素图：每个体素记录点的质心和看见它的帧数；被足够多帧看见的体素算静止。
/// 建满 build_seconds 后冻结，冻结后才供配准使用，内容不再变化。
class VenueLayer {
public:
    explicit VenueLayer(VenueLayerConfig config = {}) : config_(config) {}

    /// 插入一帧 map 系的场外点；冻结后忽略。返回这次插入是否让它冻结。
    bool insert(const Cloud& points_map, double timestamp, std::uint64_t frame_id);
    /// 是否已冻结且静止体素足够多。
    bool ready() const { return frozen_ && cloud_ && cloud_->size() >= config_.min_voxels; }
    bool frozen() const { return frozen_; }
    /// 冻结时生成的静止体素质心（map 系）；冻结前为空。
    Cloud::ConstPtr cloud() const { return cloud_; }
    /// 冻结时的版本号，合并目标按它缓存。
    std::uint64_t revision() const { return revision_; }
    std::size_t voxels() const { return cells_.size(); }
    std::uint64_t frames() const { return frames_; }

private:
    struct Cell {
        Eigen::Vector3d sum = Eigen::Vector3d::Zero();
        int points = 0;
        int frames = 0;
        std::uint64_t last_frame = 0;
    };
    void freeze();

    VenueLayerConfig config_;
    std::unordered_map<std::int64_t, Cell> cells_;
    double first_timestamp_ = -1.0;
    std::uint64_t frames_ = 0;
    bool frozen_ = false;
    Cloud::Ptr cloud_;
    std::uint64_t revision_ = 0;
};

/// 配准目标（参考关键帧系的先验子图）并上参考关键帧 radius 内的场外层点。场外层冻结后内容不变，
/// 合并结果按（关键帧编号，场外层版本）缓存，点云身份稳定，配准的目标准备缓存也能命中。
class VenueTargets {
public:
    explicit VenueTargets(std::size_t capacity) : capacity_(capacity) {}

    Cloud::ConstPtr merged(int index, std::uint64_t revision, const Cloud& target, const Cloud& venue_map,
                           const Eigen::Matrix4d& reference_pose, double radius);

private:
    std::size_t capacity_;
    std::mutex mutex_;
    std::map<std::pair<int, std::uint64_t>, Cloud::ConstPtr> merged_;
};
}  // namespace backend
