// Reference: https://github.com/moveit/moveit2_tutorials/tree/main/doc/tutorials/pick_and_place_with_moveit_task_constructor
// MTC pick-and-place spike for SO-ARM101. Hardcoded target pair.

#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/task_constructor/task.h>
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>

#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#if __has_include(<tf2_eigen/tf2_eigen.hpp>)
#include <tf2_eigen/tf2_eigen.hpp>
#else
#include <tf2_eigen/tf2_eigen.h>
#endif

#include <tf2_msgs/msg/tf_message.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <Eigen/Geometry>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace mtc = moveit::task_constructor;

namespace {
constexpr char kNodeName[] = "so_arm101_mtc";
constexpr char kRunService[] = "/so_arm101_mtc/run";
constexpr char kObjectsTopic[] = "/objects_poses_sim";
constexpr char kDropsTopic[] = "/drop_poses";

// Hardcoded targets for spike. Second iteration adds selection via action goal.
constexpr char kTargetLegoId[] = "green_2x3";
constexpr char kTargetCupId[] = "drop_0";

// Planning frame + robot frames (from so_arm101.srdf / kinematics.yaml)
constexpr char kPlanningFrame[] = "base";
constexpr char kTcpFrame[] = "tcp_link";
constexpr char kArmGroup[] = "arm";
constexpr char kGripperGroup[] = "gripper";
// End-effector NAME from SRDF (<end_effector name="endeffector" ...>).
// Distinct from the gripper group name; MTC uses this for Pick/Place eef property.
constexpr char kEefName[] = "endeffector";

// Lego bbox fallback (m). Exact bbox is in /objects_bbox_sim but for the spike
// a slight overestimate is fine — MTC only uses this for the swept volume.
constexpr double kLegoSizeX = 0.025;
constexpr double kLegoSizeY = 0.025;
constexpr double kLegoSizeZ = 0.020;

// Drop offset above cup body center — matches DROP_OFFSET_ABOVE_CUP_M in control_gui.py:3063
constexpr double kDropOffsetAboveCup = 0.10;

struct Pose7 {
  double x, y, z;
  double qx, qy, qz, qw;
};
}  // namespace

class MtcPickPlaceNode {
public:
  explicit MtcPickPlaceNode(const rclcpp::NodeOptions& options)
      : node_(std::make_shared<rclcpp::Node>(kNodeName, options)) {
    object_sub_ = node_->create_subscription<tf2_msgs::msg::TFMessage>(
        kObjectsTopic, rclcpp::SensorDataQoS(),
        [this](tf2_msgs::msg::TFMessage::SharedPtr msg) { on_tf(object_poses_, *msg); });

    drop_sub_ = node_->create_subscription<tf2_msgs::msg::TFMessage>(
        kDropsTopic, rclcpp::SensorDataQoS(),
        [this](tf2_msgs::msg::TFMessage::SharedPtr msg) { on_tf(drop_poses_, *msg); });

    run_srv_ = node_->create_service<std_srvs::srv::Trigger>(
        kRunService,
        [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
               std::shared_ptr<std_srvs::srv::Trigger::Response> res) { on_run(req, res); });

    RCLCPP_INFO(node_->get_logger(), "MTC spike ready. service=%s target=(%s -> %s)",
                kRunService, kTargetLegoId, kTargetCupId);
  }

  rclcpp::node_interfaces::NodeBaseInterface::SharedPtr base_interface() {
    return node_->get_node_base_interface();
  }

private:
  void on_tf(std::unordered_map<std::string, Pose7>& dst, const tf2_msgs::msg::TFMessage& msg) {
    std::lock_guard<std::mutex> lk(poses_mtx_);
    for (const auto& t : msg.transforms) {
      Pose7 p{
          t.transform.translation.x, t.transform.translation.y, t.transform.translation.z,
          t.transform.rotation.x,    t.transform.rotation.y,    t.transform.rotation.z,
          t.transform.rotation.w,
      };
      dst[t.child_frame_id] = p;
    }
  }

  bool lookup_pose(const std::unordered_map<std::string, Pose7>& src, const std::string& id,
                   Pose7& out) {
    std::lock_guard<std::mutex> lk(poses_mtx_);
    auto it = src.find(id);
    if (it == src.end()) return false;
    out = it->second;
    return true;
  }

  void on_run(const std::shared_ptr<std_srvs::srv::Trigger::Request>&,
              std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
    if (running_.exchange(true)) {
      res->success = false;
      res->message = "MTC task already running";
      return;
    }

    Pose7 lego_pose{}, cup_pose{};
    if (!lookup_pose(object_poses_, kTargetLegoId, lego_pose)) {
      running_ = false;
      res->success = false;
      res->message = std::string("Lego not in /objects_poses_sim: ") + kTargetLegoId;
      RCLCPP_ERROR(node_->get_logger(), "%s", res->message.c_str());
      return;
    }
    if (!lookup_pose(drop_poses_, kTargetCupId, cup_pose)) {
      running_ = false;
      res->success = false;
      res->message = std::string("Cup not in /drop_poses: ") + kTargetCupId;
      RCLCPP_ERROR(node_->get_logger(), "%s", res->message.c_str());
      return;
    }

    RCLCPP_INFO(node_->get_logger(),
                "Planning MTC pick+place: lego=(%.3f,%.3f,%.3f) cup=(%.3f,%.3f,%.3f)",
                lego_pose.x, lego_pose.y, lego_pose.z, cup_pose.x, cup_pose.y, cup_pose.z);

    spawn_lego_collision(lego_pose);

    bool ok = false;
    std::string msg;
    try {
      ok = plan_and_execute(lego_pose, cup_pose, msg);
    } catch (const std::exception& e) {
      msg = std::string("exception: ") + e.what();
      RCLCPP_ERROR(node_->get_logger(), "%s", msg.c_str());
    }

    running_ = false;
    res->success = ok;
    res->message = ok ? "MTC pick+place completed" : msg;
  }

  void spawn_lego_collision(const Pose7& lego_pose) {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = kTargetLegoId;
    obj.header.frame_id = kPlanningFrame;
    shape_msgs::msg::SolidPrimitive box;
    box.type = shape_msgs::msg::SolidPrimitive::BOX;
    box.dimensions = {kLegoSizeX, kLegoSizeY, kLegoSizeZ};
    obj.primitives.push_back(box);
    geometry_msgs::msg::Pose pose;
    pose.position.x = lego_pose.x;
    pose.position.y = lego_pose.y;
    pose.position.z = lego_pose.z;
    pose.orientation.x = lego_pose.qx;
    pose.orientation.y = lego_pose.qy;
    pose.orientation.z = lego_pose.qz;
    pose.orientation.w = lego_pose.qw;
    obj.primitive_poses.push_back(pose);
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;
    moveit::planning_interface::PlanningSceneInterface psi;
    psi.applyCollisionObject(obj);
    RCLCPP_INFO(node_->get_logger(), "Spawned lego collision object '%s'", kTargetLegoId);
  }

  mtc::Task build_task(const Pose7& lego_pose, const Pose7& cup_pose) {
    mtc::Task task;
    task.stages()->setName("so_arm101 pick place");
    task.loadRobotModel(node_);

    task.setProperty("group", kArmGroup);
    task.setProperty("eef", kEefName);
    task.setProperty("ik_frame", kTcpFrame);

    auto sampling = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    sampling->setPlannerId("RRTConnectkConfigDefault");
    auto interp = std::make_shared<mtc::solvers::JointInterpolationPlanner>();
    auto cart = std::make_shared<mtc::solvers::CartesianPath>();
    cart->setMaxVelocityScalingFactor(0.5);
    cart->setMaxAccelerationScalingFactor(0.5);
    cart->setStepSize(0.01);

    mtc::Stage* current_state_ptr = nullptr;
    mtc::Stage* attach_object_stage = nullptr;

    {
      auto stage = std::make_unique<mtc::stages::CurrentState>("current");
      current_state_ptr = stage.get();
      task.add(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::MoveTo>("open gripper", interp);
      stage->setGroup(kGripperGroup);
      stage->setGoal("open");
      task.add(std::move(stage));
    }

    {
      auto stage = std::make_unique<mtc::stages::Connect>(
          "move to pick",
          mtc::stages::Connect::GroupPlannerVector{{kArmGroup, sampling}});
      stage->setTimeout(5.0);
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      task.add(std::move(stage));
    }

    // Pick container
    {
      auto pick = std::make_unique<mtc::SerialContainer>("pick");
      task.properties().exposeTo(pick->properties(), {"eef", "group", "ik_frame"});
      pick->properties().configureInitFrom(mtc::Stage::PARENT,
                                           {"eef", "group", "ik_frame"});

      {
        // Allow gripper<->lego collisions BEFORE IK so IK solver doesn't
        // reject states where tcp_link is near lego (we have no tcp_offset
        // like Panda's 0.1m hand→tip distance).
        auto stage =
            std::make_unique<mtc::stages::ModifyPlanningScene>("allow gripper<->lego");
        stage->allowCollisions(kTargetLegoId,
                               task.getRobotModel()
                                   ->getJointModelGroup(kGripperGroup)
                                   ->getLinkModelNamesWithCollisionGeometry(),
                               true);
        pick->insert(std::move(stage));
      }

      {
        auto stage = std::make_unique<mtc::stages::MoveRelative>("approach", cart);
        stage->properties().set("marker_ns", "approach");
        stage->properties().set("link", kTcpFrame);
        stage->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        stage->setMinMaxDistance(0.02, 0.08);
        geometry_msgs::msg::Vector3Stamped vec;
        vec.header.frame_id = kPlanningFrame;
        vec.vector.z = -1.0;  // approach from above
        stage->setDirection(vec);
        pick->insert(std::move(stage));
      }

      {
        // Hardcoded single grasp pose: slightly above lego center so tcp_link
        // hovers at lego top surface rather than inside the collision box.
        // Cartesian approach above brings tcp_link down into position.
        auto gen = std::make_unique<mtc::stages::GeneratePose>("grasp pose");
        geometry_msgs::msg::PoseStamped grasp_ps;
        grasp_ps.header.frame_id = kPlanningFrame;
        grasp_ps.pose.position.x = lego_pose.x;
        grasp_ps.pose.position.y = lego_pose.y;
        grasp_ps.pose.position.z = lego_pose.z + 0.5 * kLegoSizeZ;
        // Top-down: rotate 180deg about X (world) so tcp_link -Z aligns with world -Z
        Eigen::Quaterniond q =
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()) * Eigen::Quaterniond::Identity();
        grasp_ps.pose.orientation.x = q.x();
        grasp_ps.pose.orientation.y = q.y();
        grasp_ps.pose.orientation.z = q.z();
        grasp_ps.pose.orientation.w = q.w();
        gen->setPose(grasp_ps);
        gen->setMonitoredStage(current_state_ptr);

        auto ik = std::make_unique<mtc::stages::ComputeIK>("grasp IK", std::move(gen));
        ik->setMaxIKSolutions(4);
        ik->setMinSolutionDistance(0.1);
        ik->setIKFrame(kTcpFrame);
        ik->properties().configureInitFrom(mtc::Stage::PARENT, {"eef", "group"});
        ik->properties().configureInitFrom(mtc::Stage::INTERFACE, {"target_pose"});
        pick->insert(std::move(ik));
      }

      {
        auto stage =
            std::make_unique<mtc::stages::ModifyPlanningScene>("allow gripper<->lego");
        stage->allowCollisions(kTargetLegoId,
                               task.getRobotModel()
                                   ->getJointModelGroup(kGripperGroup)
                                   ->getLinkModelNamesWithCollisionGeometry(),
                               true);
        pick->insert(std::move(stage));
      }

      {
        auto stage = std::make_unique<mtc::stages::MoveTo>("close gripper", interp);
        stage->setGroup(kGripperGroup);
        stage->setGoal("close");
        pick->insert(std::move(stage));
      }

      {
        auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("attach lego");
        stage->attachObject(kTargetLegoId, kTcpFrame);
        attach_object_stage = stage.get();
        pick->insert(std::move(stage));
      }

      {
        auto stage = std::make_unique<mtc::stages::MoveRelative>("lift", cart);
        stage->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        stage->setMinMaxDistance(0.03, 0.10);
        stage->setIKFrame(kTcpFrame);
        stage->properties().set("marker_ns", "lift");
        geometry_msgs::msg::Vector3Stamped vec;
        vec.header.frame_id = kPlanningFrame;
        vec.vector.z = 1.0;
        stage->setDirection(vec);
        pick->insert(std::move(stage));
      }

      task.add(std::move(pick));
    }

    {
      auto stage = std::make_unique<mtc::stages::Connect>(
          "move to place",
          mtc::stages::Connect::GroupPlannerVector{{kArmGroup, sampling}});
      stage->setTimeout(5.0);
      stage->properties().configureInitFrom(mtc::Stage::PARENT);
      task.add(std::move(stage));
    }

    // Place container
    {
      auto place = std::make_unique<mtc::SerialContainer>("place");
      task.properties().exposeTo(place->properties(), {"eef", "group", "ik_frame"});
      place->properties().configureInitFrom(mtc::Stage::PARENT,
                                            {"eef", "group", "ik_frame"});

      {
        auto gen = std::make_unique<mtc::stages::GeneratePlacePose>("drop pose");
        gen->setObject(kTargetLegoId);
        gen->setMonitoredStage(attach_object_stage);
        geometry_msgs::msg::PoseStamped drop_ps;
        drop_ps.header.frame_id = kPlanningFrame;
        drop_ps.pose.position.x = cup_pose.x;
        drop_ps.pose.position.y = cup_pose.y;
        drop_ps.pose.position.z = cup_pose.z + kDropOffsetAboveCup;
        drop_ps.pose.orientation.w = 1.0;
        gen->setPose(drop_ps);

        auto ik = std::make_unique<mtc::stages::ComputeIK>("drop IK", std::move(gen));
        ik->setMaxIKSolutions(4);
        ik->setMinSolutionDistance(0.1);
        ik->setIKFrame(kTargetLegoId);
        ik->properties().configureInitFrom(mtc::Stage::PARENT, {"eef", "group"});
        ik->properties().configureInitFrom(mtc::Stage::INTERFACE, {"target_pose"});
        place->insert(std::move(ik));
      }

      {
        auto stage = std::make_unique<mtc::stages::MoveTo>("open gripper", interp);
        stage->setGroup(kGripperGroup);
        stage->setGoal("open");
        place->insert(std::move(stage));
      }

      {
        auto stage =
            std::make_unique<mtc::stages::ModifyPlanningScene>("forbid gripper<->lego");
        stage->allowCollisions(kTargetLegoId,
                               task.getRobotModel()
                                   ->getJointModelGroup(kGripperGroup)
                                   ->getLinkModelNamesWithCollisionGeometry(),
                               false);
        place->insert(std::move(stage));
      }

      {
        auto stage = std::make_unique<mtc::stages::ModifyPlanningScene>("detach lego");
        stage->detachObject(kTargetLegoId, kTcpFrame);
        place->insert(std::move(stage));
      }

      {
        auto stage = std::make_unique<mtc::stages::MoveRelative>("retreat", cart);
        stage->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        stage->setMinMaxDistance(0.03, 0.10);
        stage->setIKFrame(kTcpFrame);
        stage->properties().set("marker_ns", "retreat");
        geometry_msgs::msg::Vector3Stamped vec;
        vec.header.frame_id = kPlanningFrame;
        vec.vector.z = 1.0;
        stage->setDirection(vec);
        place->insert(std::move(stage));
      }

      task.add(std::move(place));
    }

    return task;
  }

  bool plan_and_execute(const Pose7& lego_pose, const Pose7& cup_pose, std::string& out_msg) {
    auto task = build_task(lego_pose, cup_pose);
    try {
      task.init();
    } catch (mtc::InitStageException& e) {
      out_msg = std::string("init failed: ") + e.what();
      RCLCPP_ERROR_STREAM(node_->get_logger(), e);
      return false;
    }

    if (!task.plan(5)) {
      out_msg = "planning failed (no solution found)";
      RCLCPP_ERROR(node_->get_logger(), "%s", out_msg.c_str());
      // Dump per-stage solution/failure counts to locate which stage ran dry.
      task.stages()->traverseRecursively([this](const mtc::Stage& s, unsigned int depth) {
        std::string indent(2 * depth, ' ');
        RCLCPP_ERROR(node_->get_logger(),
                     "%s[stage] %s — solutions=%zu failures=%zu",
                     indent.c_str(), s.name().c_str(), s.solutions().size(),
                     s.failures().size());
        return true;
      });
      return false;
    }

    task.introspection().publishSolution(*task.solutions().front());
    auto result = task.execute(*task.solutions().front());
    if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      out_msg = "execution failed: MoveItErrorCode=" + std::to_string(result.val);
      RCLCPP_ERROR(node_->get_logger(), "%s", out_msg.c_str());
      return false;
    }

    out_msg = "ok";
    return true;
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr object_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr drop_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr run_srv_;

  std::mutex poses_mtx_;
  std::unordered_map<std::string, Pose7> object_poses_;
  std::unordered_map<std::string, Pose7> drop_poses_;

  std::atomic<bool> running_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<MtcPickPlaceNode>(options);
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node->base_interface());
  executor.spin();
  executor.remove_node(node->base_interface());
  rclcpp::shutdown();
  return 0;
}
