/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/internal/2d/local_trajectory_builder_2d.h"

#include <limits>
#include <memory>

#include "absl/memory/memory.h"
#include "cartographer/metrics/family_factory.h"
#include "cartographer/sensor/range_data.h"

namespace cartographer {
namespace mapping {

static auto* kLocalSlamLatencyMetric = metrics::Gauge::Null();
static auto* kLocalSlamRealTimeRatio = metrics::Gauge::Null();
static auto* kLocalSlamCpuRealTimeRatio = metrics::Gauge::Null();
static auto* kRealTimeCorrelativeScanMatcherScoreMetric =
        metrics::Histogram::Null();
static auto* kCeresScanMatcherCostMetric = metrics::Histogram::Null();
static auto* kScanMatcherResidualDistanceMetric = metrics::Histogram::Null();
static auto* kScanMatcherResidualAngleMetric = metrics::Histogram::Null();

LocalTrajectoryBuilder2D::LocalTrajectoryBuilder2D(
        const proto::LocalTrajectoryBuilderOptions2D& options,
        const std::vector<std::string>& expected_range_sensor_ids)
    : options_(options),
      active_submaps_(options.submaps_options()),
      motion_filter_(options_.motion_filter_options()),
      real_time_correlative_scan_matcher_(
          options_.real_time_correlative_scan_matcher_options()),
      ceres_scan_matcher_(options_.ceres_scan_matcher_options()),
      range_data_collator_(expected_range_sensor_ids) {}

LocalTrajectoryBuilder2D::~LocalTrajectoryBuilder2D() {}

//把数据转换到重力对齐的坐标系,即水平方向.
//并进行网格滤波
sensor::RangeData LocalTrajectoryBuilder2D::TransformToGravityAlignedFrameAndFilter(
        const transform::Rigid3f& transform_to_gravity_aligned_frame,
        const sensor::RangeData& range_data) const
{
    //把点云投影到平面上来.
    const sensor::RangeData cropped =
            sensor::CropRangeData(sensor::TransformRangeData(
                                      range_data, transform_to_gravity_aligned_frame),
                                  options_.min_z(), options_.max_z());

    //对于投影到平面的点云进行网格滤波,去除冗余的点.
    return sensor::RangeData
    {
        cropped.origin,
                sensor::VoxelFilter(options_.voxel_filter_size()).Filter(cropped.returns),
                sensor::VoxelFilter(options_.voxel_filter_size()).Filter(cropped.misses)
    };
}

//进行scanMatch，CSM+Ceres
std::unique_ptr<transform::Rigid2d> LocalTrajectoryBuilder2D::ScanMatch(
        const common::Time time,
        const transform::Rigid2d& pose_prediction,
        const sensor::PointCloud& filtered_gravity_aligned_point_cloud)
{
    if (active_submaps_.submaps().empty())
    {
        return absl::make_unique<transform::Rigid2d>(pose_prediction);
    }

    //进行匹配的子图
    std::shared_ptr<const Submap2D> matching_submap =
            active_submaps_.submaps().front();

    // The online correlative scan matcher will refine the initial estimate for
    // the Ceres scan matcher.
    // 初始的位姿(使用之前位置插值器预测的位姿)
    transform::Rigid2d initial_ceres_pose = pose_prediction;

    //是否进行real-time csm的匹配.使用real-time csm的匹配更新initial_ceres_pose，并用于后面的ceres_scan_matcher
    if (options_.use_online_correlative_scan_matching())
    {
        const double score = real_time_correlative_scan_matcher_.Match(
                    pose_prediction,
                    filtered_gravity_aligned_point_cloud,
                    *matching_submap->grid(),
                    &initial_ceres_pose);

        kRealTimeCorrelativeScanMatcherScoreMetric->Observe(score);
    }

    //进行ceres的匹配.--即基于优化的方法来进行匹配.
    auto pose_observation = absl::make_unique<transform::Rigid2d>();
    ceres::Solver::Summary summary;
    ceres_scan_matcher_.Match(pose_prediction.translation(), initial_ceres_pose,
                              filtered_gravity_aligned_point_cloud,
                              *matching_submap->grid(), pose_observation.get(),
                              &summary);

    if (pose_observation)
    {
        kCeresScanMatcherCostMetric->Observe(summary.final_cost);

        const double residual_distance =
                (pose_observation->translation() - pose_prediction.translation())
                .norm();

        kScanMatcherResidualDistanceMetric->Observe(residual_distance);

        const double residual_angle =
                std::abs(pose_observation->rotation().angle() -
                         pose_prediction.rotation().angle());

        kScanMatcherResidualAngleMetric->Observe(residual_angle);
    }
    return pose_observation;
}

//增加激光数的函数,本函数主要
//1.多传感器数据同步.
//2.运动畸变去除.
//3.无效数据滤除.
//4.调用AddAccumulatedRangeData()函数实现.

std::unique_ptr<LocalTrajectoryBuilder2D::MatchingResult>
LocalTrajectoryBuilder2D::AddRangeData(
        const std::string& sensor_id,
        const sensor::TimedPointCloudData& unsynchronized_data)
{
    //进行多个传感器的数据同步,只有一个传感器的时候,可以直接忽略
    auto synchronized_data =
            range_data_collator_.AddRangeData(sensor_id, unsynchronized_data);

    if (synchronized_data.ranges.empty())
    {
        LOG(INFO) << "Range data collator filling buffer.";
        return nullptr;
    }

    //数据的时间戳
    const common::Time& time = synchronized_data.time;

    // Initialize extrapolator now if we do not ever use an IMU.
    if (!options_.use_imu_data())
    {
        InitializeExtrapolator(time);
    }

    if (extrapolator_ == nullptr)
    {
        // Until we've initialized the extrapolator with our first IMU message, we
        // cannot compute the orientation of the rangefinder.
        LOG(INFO) << "Extrapolator not yet initialized.";
        return nullptr;
    }

    CHECK(!synchronized_data.ranges.empty());
    // TODO(gaschler): Check if this can strictly be 0.
    CHECK_LE(synchronized_data.ranges.back().point_time.time, 0.f);

    //第一个激光束的时间戳.
    const common::Time time_first_point =
            time +
            common::FromSeconds(synchronized_data.ranges.front().point_time.time);

    if (time_first_point < extrapolator_->GetLastPoseTime())
    {
        LOG(INFO) << "Extrapolator is still initializing.";
        return nullptr;
    }

    //插值得到每一个点的位姿，相当于运行畸变去除．
    std::vector<transform::Rigid3f> range_data_poses;
    range_data_poses.reserve(synchronized_data.ranges.size());
    bool warned = false;

    //枚举每一个激光点.
    for (const auto& range : synchronized_data.ranges)
    {
        //每一个激光束是时间戳.
        common::Time time_point = time + common::FromSeconds(range.point_time.time);

        if (time_point < extrapolator_->GetLastExtrapolatedTime())
        {
            if (!warned)
            {
                LOG(ERROR)
                        << "Timestamp of individual range data point jumps backwards from "
                        << extrapolator_->GetLastExtrapolatedTime() << " to " << time_point;
                warned = true;
            }

            //根据时间戳进行位姿差值.
            time_point = extrapolator_->GetLastExtrapolatedTime();
        }

        //每一个激光点的位姿.
        range_data_poses.push_back(
                    extrapolator_->ExtrapolatePose(time_point).cast<float>());//根据时间插值每一个位置
    }

    if (num_accumulated_ == 0)
    {
        // 'accumulated_range_data_.origin' is uninitialized until the last
        // accumulation.
        accumulated_range_data_ = sensor::RangeData{{}, {}, {}};
    }

    // Drop any returns below the minimum range and convert returns beyond the
    // maximum range into misses.
    // 得到所有的累计数据，这里进行运动畸变去除．
    for (size_t i = 0; i < synchronized_data.ranges.size(); ++i)
    {
        const sensor::TimedRangefinderPoint& hit =
                synchronized_data.ranges[i].point_time;

        //原点的位姿
        const Eigen::Vector3f origin_in_local =
                range_data_poses[i] *
                synchronized_data.origins.at(synchronized_data.ranges[i].origin_index);

        //击中点的位姿.
        sensor::RangefinderPoint hit_in_local =
                range_data_poses[i] * sensor::ToRangefinderPoint(hit);

        //激光束的距离
        const Eigen::Vector3f delta = hit_in_local.position - origin_in_local;
        const float range = delta.norm();

        //根据范围判断是否合法.根据距离判断是hit还是miss.hit是打中障碍物的光束，miss是没有打中障碍物的光束
        if (range >= options_.min_range())
        {
            if (range <= options_.max_range())
            {
                accumulated_range_data_.returns.push_back(hit_in_local);
            }
            else
            {
                hit_in_local.position =
                        origin_in_local +
                        options_.missing_data_ray_length() / range * delta;
                accumulated_range_data_.misses.push_back(hit_in_local);
            }
        }
    }

    //累计数据自加
    ++num_accumulated_;

    //如果累计的数据足够，则进行匹配．默认是1帧匹配一次
    if (num_accumulated_ >= options_.num_accumulated_range_data())
    {
        //计算时间差.
        const common::Time current_sensor_time = synchronized_data.time;
        absl::optional<common::Duration> sensor_duration;
        if (last_sensor_time_.has_value())
        {
            sensor_duration = current_sensor_time - last_sensor_time_.value();
        }

        last_sensor_time_ = current_sensor_time;
        num_accumulated_ = 0;

        //得到当前时刻的重力向量.把激光投影到平面来。
        const transform::Rigid3d gravity_alignment = transform::Rigid3d::Rotation(
                    extrapolator_->EstimateGravityOrientation(time));

        // TODO(gaschler): This assumes that 'range_data_poses.back()' is at time
        // 'time'.
        // 累计数据的原点.
        accumulated_range_data_.origin = range_data_poses.back().translation();

        //最终调用的函数. 1）将激光数据进行运动畸变去除。2）然后根据重力向量将数据点投影到平面。3）调用下面的函数进行scanMatch。
        return AddAccumulatedRangeData(
                    time,
                    TransformToGravityAlignedFrameAndFilter(//将数据转换到平面并进行网格稀疏化(Voxel Filter)
                        gravity_alignment.cast<float>() * range_data_poses.back().inverse(),
                        accumulated_range_data_),
                    gravity_alignment, sensor_duration);
    }
    return nullptr;
}


//这里面会进行激光数据的匹配．
std::unique_ptr<LocalTrajectoryBuilder2D::MatchingResult>
LocalTrajectoryBuilder2D::AddAccumulatedRangeData(
        const common::Time time,
        const sensor::RangeData& gravity_aligned_range_data,//这个参数传进来的是经过体素过滤器滤波的
        const transform::Rigid3d& gravity_alignment,
        const absl::optional<common::Duration>& sensor_duration)
{
    if (gravity_aligned_range_data.returns.empty())
    {
        LOG(WARNING) << "Dropped empty horizontal range data.";
        return nullptr;
    }

    // Computes a gravity aligned pose prediction.
    // 预测6自由度的位姿
    const transform::Rigid3d non_gravity_aligned_pose_prediction =
            extrapolator_->ExtrapolatePose(time);

    // 把上面的6自由度位姿投影成3自由度的位姿．--作为scan-match的初始解.
    const transform::Rigid2d pose_prediction = transform::Project2D(
                non_gravity_aligned_pose_prediction * gravity_alignment.inverse());

    //进行自适应网格滤波滤波．(Adaptive Voxel Filter)
    const sensor::PointCloud& filtered_gravity_aligned_point_cloud =
            sensor::AdaptiveVoxelFilter(options_.adaptive_voxel_filter_options())
            .Filter(gravity_aligned_range_data.returns);

    if (filtered_gravity_aligned_point_cloud.empty())
    {
        return nullptr;
    }

    // local map frame <- gravity-aligned frame
    // 进行ScanMatch匹配．得到帧间匹配的位姿(Scan Matching)
    std::unique_ptr<transform::Rigid2d> pose_estimate_2d =
            ScanMatch(time, pose_prediction, filtered_gravity_aligned_point_cloud);//pose_prediction为用位置插值器预测的位姿

    if (pose_estimate_2d == nullptr)
    {
        LOG(WARNING) << "Scan matching failed.";
        return nullptr;
    }

    //把位姿重新换回到６自由度．
    const transform::Rigid3d pose_estimate =
            transform::Embed3D(*pose_estimate_2d) * gravity_alignment;

    //更新估计器.
    extrapolator_->AddPose(time, pose_estimate);

    //把点云转换到估计出来的位姿中．即Submap坐标系中．
    sensor::RangeData range_data_in_local =
            TransformRangeData(gravity_aligned_range_data,
                               transform::Embed3D(pose_estimate_2d->cast<float>()));

    //把点云插入到SubMap中,返回插入结果.
    std::unique_ptr<InsertionResult> insertion_result = InsertIntoSubmap(
                time, range_data_in_local, filtered_gravity_aligned_point_cloud,
                pose_estimate, gravity_alignment.rotation());

    const auto wall_time = std::chrono::steady_clock::now();
    if (last_wall_time_.has_value())
    {
        const auto wall_time_duration = wall_time - last_wall_time_.value();
        kLocalSlamLatencyMetric->Set(common::ToSeconds(wall_time_duration));
        if (sensor_duration.has_value())
        {
            kLocalSlamRealTimeRatio->Set(common::ToSeconds(sensor_duration.value()) /
                                         common::ToSeconds(wall_time_duration));
        }
    }

    const double thread_cpu_time_seconds = common::GetThreadCpuTimeSeconds();
    if (last_thread_cpu_time_seconds_.has_value())
    {
        const double thread_cpu_duration_seconds =
                thread_cpu_time_seconds - last_thread_cpu_time_seconds_.value();
        if (sensor_duration.has_value())
        {
            kLocalSlamCpuRealTimeRatio->Set(
                        common::ToSeconds(sensor_duration.value()) /
                        thread_cpu_duration_seconds);
        }
    }

    last_wall_time_ = wall_time;
    last_thread_cpu_time_seconds_ = thread_cpu_time_seconds;
    return absl::make_unique<MatchingResult>(
                MatchingResult{time, pose_estimate, std::move(range_data_in_local),
                               std::move(insertion_result)});//将scan matching的结果打包返回
}

//插入到局部子图中．(Motion Filter Still)
std::unique_ptr<LocalTrajectoryBuilder2D::InsertionResult>
LocalTrajectoryBuilder2D::InsertIntoSubmap(
        const common::Time time, const sensor::RangeData& range_data_in_local,
        const sensor::PointCloud& filtered_gravity_aligned_point_cloud,
        const transform::Rigid3d& pose_estimate,
        const Eigen::Quaterniond& gravity_alignment)
{
    //如果位移，旋转的幅度小于设定的阈值就直接丢掉该匹配得到的位姿
    if (motion_filter_.IsSimilar(time, pose_estimate))
    {
        return nullptr;
    }

    //把当前帧数据,插入到地图中.
    std::vector<std::shared_ptr<const Submap2D>> insertion_submaps =
            active_submaps_.InsertRangeData(range_data_in_local);

    return absl::make_unique<InsertionResult>(InsertionResult{
                                                  std::make_shared<const TrajectoryNode::Data>(TrajectoryNode::Data{
                                                      time,
                                                      gravity_alignment,
                                                      filtered_gravity_aligned_point_cloud,
                                                      {},  // 'high_resolution_point_cloud' is only used in 3D.
                                                      {},  // 'low_resolution_point_cloud' is only used in 3D.
                                                      {},  // 'rotational_scan_matcher_histogram' is only used in 3D.
                                                      pose_estimate}),
                                                  std::move(insertion_submaps)});
}

//加入IMU数据,用来初始化和更新PoseExtrapolator
void LocalTrajectoryBuilder2D::AddImuData(const sensor::ImuData& imu_data)
{
    CHECK(options_.use_imu_data()) << "An unexpected IMU packet was added.";
    InitializeExtrapolator(imu_data.time);
    extrapolator_->AddImuData(imu_data);
}

//加入里程计数据,用来初始化更新PoseExtrapolator
void LocalTrajectoryBuilder2D::AddOdometryData(
        const sensor::OdometryData& odometry_data)
{
    if (extrapolator_ == nullptr)
    {
        // Until we've initialized the extrapolator we cannot add odometry data.
        LOG(INFO) << "Extrapolator not yet initialized.";
        return;
    }

    extrapolator_->AddOdometryData(odometry_data);
}

//初始化PoseExtrapolator.
void LocalTrajectoryBuilder2D::InitializeExtrapolator(const common::Time time)
{
    if (extrapolator_ != nullptr)
    {
        return;
    }

    // We derive velocities from poses which are at least 1 ms apart for numerical
    // stability. Usually poses known to the extrapolator will be further apart
    // in time and thus the last two are used.
    // 用相差至少1ms的两个位姿来计算速度.
    constexpr double kExtrapolationEstimationTimeSec = 0.001;

    // TODO(gaschler): Consider using InitializeWithImu as 3D does.
    extrapolator_ = absl::make_unique<PoseExtrapolator>(
                ::cartographer::common::FromSeconds(kExtrapolationEstimationTimeSec),
                options_.imu_gravity_time_constant());

    extrapolator_->AddPose(time, transform::Rigid3d::Identity());
}

void LocalTrajectoryBuilder2D::RegisterMetrics(
        metrics::FamilyFactory* family_factory) {
    auto* latency = family_factory->NewGaugeFamily(
                "mapping_2d_local_trajectory_builder_latency",
                "Duration from first incoming point cloud in accumulation to local slam "
                "result");
    kLocalSlamLatencyMetric = latency->Add({});
    auto* real_time_ratio = family_factory->NewGaugeFamily(
                "mapping_2d_local_trajectory_builder_real_time_ratio",
                "sensor duration / wall clock duration.");
    kLocalSlamRealTimeRatio = real_time_ratio->Add({});

    auto* cpu_real_time_ratio = family_factory->NewGaugeFamily(
                "mapping_2d_local_trajectory_builder_cpu_real_time_ratio",
                "sensor duration / cpu duration.");
    kLocalSlamCpuRealTimeRatio = cpu_real_time_ratio->Add({});
    auto score_boundaries = metrics::Histogram::FixedWidth(0.05, 20);
    auto* scores = family_factory->NewHistogramFamily(
                "mapping_2d_local_trajectory_builder_scores", "Local scan matcher scores",
                score_boundaries);
    kRealTimeCorrelativeScanMatcherScoreMetric =
            scores->Add({{"scan_matcher", "real_time_correlative"}});
    auto cost_boundaries = metrics::Histogram::ScaledPowersOf(2, 0.01, 100);
    auto* costs = family_factory->NewHistogramFamily(
                "mapping_2d_local_trajectory_builder_costs", "Local scan matcher costs",
                cost_boundaries);
    kCeresScanMatcherCostMetric = costs->Add({{"scan_matcher", "ceres"}});
    auto distance_boundaries = metrics::Histogram::ScaledPowersOf(2, 0.01, 10);
    auto* residuals = family_factory->NewHistogramFamily(
                "mapping_2d_local_trajectory_builder_residuals",
                "Local scan matcher residuals", distance_boundaries);
    kScanMatcherResidualDistanceMetric =
            residuals->Add({{"component", "distance"}});
    kScanMatcherResidualAngleMetric = residuals->Add({{"component", "angle"}});
}

}  // namespace mapping
}  // namespace cartographer
