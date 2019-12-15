/*
 * Copyright 2017 The Cartographer Authors
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

#include "cartographer/mapping/pose_extrapolator.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

PoseExtrapolator::PoseExtrapolator(const common::Duration pose_queue_duration,
                                   double imu_gravity_time_constant)
    : pose_queue_duration_(pose_queue_duration),
      gravity_time_constant_(imu_gravity_time_constant),
      cached_extrapolated_pose_{common::Time::min(),
                                transform::Rigid3d::Identity()} {}

std::unique_ptr<PoseExtrapolator> PoseExtrapolator::InitializeWithImu(
    const common::Duration pose_queue_duration,
    const double imu_gravity_time_constant, const sensor::ImuData& imu_data)
{
  auto extrapolator = absl::make_unique<PoseExtrapolator>(
      pose_queue_duration, imu_gravity_time_constant);

  extrapolator->AddImuData(imu_data);

  extrapolator->imu_tracker_ =
      absl::make_unique<ImuTracker>(imu_gravity_time_constant, imu_data.time);

  extrapolator->imu_tracker_->AddImuLinearAccelerationObservation(
      imu_data.linear_acceleration);

  extrapolator->imu_tracker_->AddImuAngularVelocityObservation(
      imu_data.angular_velocity);

  extrapolator->imu_tracker_->Advance(imu_data.time);

  extrapolator->AddPose(
      imu_data.time,
      transform::Rigid3d::Rotation(extrapolator->imu_tracker_->orientation()));

  return extrapolator;
}

common::Time PoseExtrapolator::GetLastPoseTime() const {
  if (timed_pose_queue_.empty()) {
    return common::Time::min();
  }
  return timed_pose_queue_.back().time;
}

common::Time PoseExtrapolator::GetLastExtrapolatedTime() const {
  if (!extrapolation_imu_tracker_) {
    return common::Time::min();
  }
  return extrapolation_imu_tracker_->time();
}

//加入位姿.
void PoseExtrapolator::AddPose(const common::Time time,
                               const transform::Rigid3d& pose)
{
  if (imu_tracker_ == nullptr)
  {
    common::Time tracker_start = time;
    if (!imu_data_.empty())
    {
      tracker_start = std::min(tracker_start, imu_data_.front().time);
    }
    imu_tracker_ =
        absl::make_unique<ImuTracker>(gravity_time_constant_, tracker_start);
  }

  timed_pose_queue_.push_back(TimedPose{time, pose});
  while (timed_pose_queue_.size() > 2 &&
         timed_pose_queue_[1].time <= time - pose_queue_duration_)
  {
    timed_pose_queue_.pop_front();
  }

  //从位姿中估计速度.
  UpdateVelocitiesFromPoses();

  //更新imu-tracker信息.
  AdvanceImuTracker(time, imu_tracker_.get());

  //去除多余数据.
  TrimImuData();

  TrimOdometryData();

  odometry_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);

  extrapolation_imu_tracker_ = absl::make_unique<ImuTracker>(*imu_tracker_);
}

//加入IMU数据.
void PoseExtrapolator::AddImuData(const sensor::ImuData& imu_data)
{
  CHECK(timed_pose_queue_.empty() ||
        imu_data.time >= timed_pose_queue_.back().time);
  imu_data_.push_back(imu_data);
  TrimImuData();
}

//加入里程计数据.
void PoseExtrapolator::AddOdometryData(
    const sensor::OdometryData& odometry_data)
{
  CHECK(timed_pose_queue_.empty() ||
        odometry_data.time >= timed_pose_queue_.back().time);
  odometry_data_.push_back(odometry_data);

  TrimOdometryData();
  if (odometry_data_.size() < 2)
  {
    return;
  }

  // TODO(whess): Improve by using more than just the last two odometry poses.
  // Compute extrapolation in the tracking frame.
  const sensor::OdometryData& odometry_data_oldest = odometry_data_.front();
  const sensor::OdometryData& odometry_data_newest = odometry_data_.back();

  //首尾位姿的时间差.
  const double odometry_time_delta =
      common::ToSeconds(odometry_data_oldest.time - odometry_data_newest.time);

  //首尾位姿的差.
  const transform::Rigid3d odometry_pose_delta =
      odometry_data_newest.pose.inverse() * odometry_data_oldest.pose;

  //从里程计数据中估计角速度.
  angular_velocity_from_odometry_ =
      transform::RotationQuaternionToAngleAxisVector(
          odometry_pose_delta.rotation()) /
      odometry_time_delta;

  if (timed_pose_queue_.empty())
  {
    return;
  }

  //从里程计数据中估计线速度.
  const Eigen::Vector3d
      linear_velocity_in_tracking_frame_at_newest_odometry_time =
          odometry_pose_delta.translation() / odometry_time_delta;

  const Eigen::Quaterniond orientation_at_newest_odometry_time =
      timed_pose_queue_.back().pose.rotation() *
      ExtrapolateRotation(odometry_data_newest.time,
                          odometry_imu_tracker_.get());

  linear_velocity_from_odometry_ =
      orientation_at_newest_odometry_time *
      linear_velocity_in_tracking_frame_at_newest_odometry_time;
}

//对位姿进行插值,得到time时刻的位姿.
transform::Rigid3d PoseExtrapolator::ExtrapolatePose(const common::Time time)
{
  //位姿队列中最新的队列.
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();

  //必须要比newest_timed_pose的时间大.
  CHECK_GE(time, newest_timed_pose.time);

  if (cached_extrapolated_pose_.time != time)
  {
    //对平移进行插值.
    const Eigen::Vector3d translation =
        ExtrapolateTranslation(time) + newest_timed_pose.pose.translation();

    //对旋转进行插值.
    const Eigen::Quaterniond rotation =
        newest_timed_pose.pose.rotation() *
        ExtrapolateRotation(time, extrapolation_imu_tracker_.get());

    cached_extrapolated_pose_ =
        TimedPose{time, transform::Rigid3d{translation, rotation}};
  }

  return cached_extrapolated_pose_.pose;
}

//计算重力向量,用imu-tracker来进行计算.
Eigen::Quaterniond PoseExtrapolator::EstimateGravityOrientation(
    const common::Time time)
{
  ImuTracker imu_tracker = *imu_tracker_;
  AdvanceImuTracker(time, &imu_tracker);
  return imu_tracker.orientation();
}

//用位姿队列中的数据来进行线速度和角速度的计算.
void PoseExtrapolator::UpdateVelocitiesFromPoses()
{
  if (timed_pose_queue_.size() < 2)
  {
    // We need two poses to estimate velocities.
    return;
  }
  CHECK(!timed_pose_queue_.empty());
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();
  const auto newest_time = newest_timed_pose.time;

  const TimedPose& oldest_timed_pose = timed_pose_queue_.front();
  const auto oldest_time = oldest_timed_pose.time;

  const double queue_delta = common::ToSeconds(newest_time - oldest_time);
  if (queue_delta < common::ToSeconds(pose_queue_duration_))
  {
    LOG(WARNING) << "Queue too short for velocity estimation. Queue duration: "
                 << queue_delta << " s";
    return;
  }

  const transform::Rigid3d& newest_pose = newest_timed_pose.pose;
  const transform::Rigid3d& oldest_pose = oldest_timed_pose.pose;

  //估计线速度
  linear_velocity_from_poses_ =
      (newest_pose.translation() - oldest_pose.translation()) / queue_delta;

  //估计角速度.
  angular_velocity_from_poses_ =
      transform::RotationQuaternionToAngleAxisVector(
          oldest_pose.rotation().inverse() * newest_pose.rotation()) /
      queue_delta;
}

void PoseExtrapolator::TrimImuData()//丢掉超时的数据
{
  while (imu_data_.size() > 1 && !timed_pose_queue_.empty() &&
         imu_data_[1].time <= timed_pose_queue_.back().time)
  {
    imu_data_.pop_front();
  }
}

void PoseExtrapolator::TrimOdometryData()//丢掉超时的数据
{
  while (odometry_data_.size() > 2 && !timed_pose_queue_.empty() &&
         odometry_data_[1].time <= timed_pose_queue_.back().time)
  {
    odometry_data_.pop_front();
  }
}

//对imu-tracker进行更新,
//用imu队列中的数据进行更新,更新到time时刻
void PoseExtrapolator::AdvanceImuTracker(const common::Time time,
                                         ImuTracker* const imu_tracker) const
{
  CHECK_GE(time, imu_tracker->time());
  if (imu_data_.empty() || time < imu_data_.front().time)//进入到这里说明没有IMU
  {
    // There is no IMU data until 'time', so we advance the ImuTracker and use
    // the angular velocities from poses and fake gravity to help 2D stability.
    imu_tracker->Advance(time);
    imu_tracker->AddImuLinearAccelerationObservation(Eigen::Vector3d::UnitZ());//给一个虚假的重力信息
    imu_tracker->AddImuAngularVelocityObservation(
        odometry_data_.size() < 2 ? angular_velocity_from_poses_
                                  : angular_velocity_from_odometry_);
    return;
  }

  if (imu_tracker->time() < imu_data_.front().time)//如果有IMU就用IMU的数据
  {
    // Advance to the beginning of 'imu_data_'.
    imu_tracker->Advance(imu_data_.front().time);
  }

  auto it = std::lower_bound(
      imu_data_.begin(), imu_data_.end(), imu_tracker->time(),
      [](const sensor::ImuData& imu_data, const common::Time& time)
  {
        return imu_data.time < time;
      });

  //用imu数据队列进行imu-tracker的数据更新.
  while (it != imu_data_.end() && it->time < time)
  {
    imu_tracker->Advance(it->time);
    imu_tracker->AddImuLinearAccelerationObservation(it->linear_acceleration);
    imu_tracker->AddImuAngularVelocityObservation(it->angular_velocity);
    ++it;
  }

  imu_tracker->Advance(time);
}

//用imu-tracker进行插值--得到的是增量.
Eigen::Quaterniond PoseExtrapolator::ExtrapolateRotation(
    const common::Time time, ImuTracker* const imu_tracker) const
{
  CHECK_GE(time, imu_tracker->time());
  AdvanceImuTracker(time, imu_tracker);
  const Eigen::Quaterniond last_orientation = imu_tracker_->orientation();
  return last_orientation.inverse() * imu_tracker->orientation();
}

//用线速度进行插值.--得到的是增量.
Eigen::Vector3d PoseExtrapolator::ExtrapolateTranslation(common::Time time)
{
  //位姿队列的最新位姿.
  const TimedPose& newest_timed_pose = timed_pose_queue_.back();

  //插值的时间长度
  const double extrapolation_delta =
      common::ToSeconds(time - newest_timed_pose.time);

  if (odometry_data_.size() < 2)//里程计数据少就用pose算出的线速度来插值位移增量
  {
    return extrapolation_delta * linear_velocity_from_poses_;
  }

  return extrapolation_delta * linear_velocity_from_odometry_;
}

}  // namespace mapping
}  // namespace cartographer
