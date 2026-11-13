"""Tool table for the SO-ARM101 digital twin: one entry per command.

MCP_TOOL_REGISTRY  name -> {"description", "parameters"} (omni-kit-mcp parameter vocabulary:
                   type / description / default / required per parameter)
MCP_HANDLERS       name -> DigitalTwin method that implements it (returns a dict with "status")

Pure data, stdlib only: isaac/soarm101_tools reads this file by path to register the commands
as soarm101.<name> on the omni-kit-mcp bridge, without importing the Kit-dependent extension.
Arbitrary code execution is the bridge's built-in run_python, so it has no entry here.
"""

MCP_TOOL_REGISTRY = {
    "play_scene": {
        "description": "Start/resume the simulation timeline in Isaac Sim.",
        "parameters": {}
    },
    "stop_scene": {
        "description": "Stop the simulation timeline in Isaac Sim.",
        "parameters": {}
    },
    "sort_objects": {
        "description": "Sort lego blocks by placing same-color blocks close together. Optionally filter by color.",
        "parameters": {
            "color": {
                "type": "string",
                "description": "Optional color to sort ('red', 'green', 'blue'). If omitted, sorts all colors."
            }
        }
    },
    "randomize_object_poses": {
        "description": "Randomize lego block positions. Optionally filter by color to only randomize that color, and/or require N legos of a chosen color land inside the workspace camera's view (for VLA-style recordings where the target object must be visible to the policy).",
        "parameters": {
            "color": {
                "type": "string",
                "description": "Optional color whose blocks get NEW positions ('red', 'green', 'blue'). If omitted, randomizes all."
            },
            "required_visible_color": {
                "type": "string",
                "description": "Optional color ('red', 'green', 'blue') that MUST be in workspace-camera FOV after randomization. Sampler retries until satisfied or its retry budget is exhausted."
            },
            "required_visible_count": {
                "type": "integer",
                "description": "How many legos of required_visible_color must be in frame (default 1, capped at the count of that-color legos present)."
            },
            "camera_fov_margin_px": {
                "type": "integer",
                "description": "Pixels of headroom from any image edge for must-be-visible legos (default 30 on a 640x480 frame)."
            }
        }
    },
    "randomize_single_object": {
        "description": "Randomize the position of a single named object in a clear workspace area, keeping all other objects fixed.",
        "parameters": {
            "object_name": {
                "type": "string",
                "description": "Name of the object to randomize (e.g. 'inverted_u_yellow')"
            }
        }
    },
    "save_scene_state": {
        "description": "Saves current object poses to a JSON file so it can be retrieved later. If json_file_path is not provided, a timestamped filename is used automatically.",
        "parameters": {
            "json_file_path": {
                "type": "string",
                "description": "Optional filename for the saved JSON (e.g. 'assembled.json'). Saved inside the scene_states directory. If omitted, a timestamped name is generated."
            }
        }
    },
    "restore_scene_state": {
        "description": "Restores previously saved object poses from a JSON file to the scene. If json_file_path is not provided, the most recent save is restored.",
        "parameters": {
            "json_file_path": {
                "type": "string",
                "description": "Optional filename of the JSON to restore (e.g. 'assembled.json'). Looked up inside the scene_states directory. If omitted, the latest timestamped save is used."
            }
        }
    },
    "add_objects": {
        "description": "Add lego block objects to the scene from the SO-ARM101 objects folder.",
        "parameters": {}
    },
    "delete_objects": {
        "description": "Delete all objects from /World/Objects and the associated pose publisher graph.",
        "parameters": {}
    },
    "setup_pose_publisher": {
        "description": "Create an action graph to publish object poses to ROS2 topic 'objects_poses_sim'.",
        "parameters": {}
    },
    "sync_real_poses": {
        "description": "Subscribe to /objects_poses_real ROS2 topic and update sim object poses to match real-world detected poses.",
        "parameters": {}
    },
    "load_scene": {
        "description": "Initialize the simulation scene with physics, ground plane, and frame rate settings.",
        "parameters": {}
    },
    "load_robot": {
        "description": "Import the SO-ARM101 robot with integrated gripper into the scene. Loads USD, configures articulation, joint drives, and gripper physics material.",
        "parameters": {}
    },
    "setup_action_graph": {
        "description": "Create the ROS2 action graph for SO-ARM101 joint state subscription and clock publishing.",
        "parameters": {}
    },
    "setup_force_publisher": {
        "description": "Create the ROS2 force/torque publisher action graph for SO-ARM101 end-effector wrench.",
        "parameters": {}
    },
    "setup_bbox_publisher": {
        "description": "Publish object bounding box dimensions to ROS2 topic 'objects_bbox_sim' as JSON.",
        "parameters": {}
    },
    "setup_wrist_camera_action_graph": {
        "description": "Create ROS2 action graph for wrist camera (OV9732) RGB and camera_info publishing.",
        "parameters": {}
    },
    "add_cups": {
        "description": "Add sorting cups (red, green, blue) in an arc in front of the robot base. Cups serve as containers for color-sorted lego blocks.",
        "parameters": {}
    },
    "publish_drop_poses": {
        "description": "Create an action graph to publish ArUco marker poses for each cup to ROS2 topic '/drop_poses' (TFMessage). Each transform's child_frame_id is drop_0, drop_1, or drop_2. Requires cups to be present in the scene.",
        "parameters": {}
    },
    "delete_cups": {
        "description": "Delete all sorting cups from /World/Containers.",
        "parameters": {}
    },
    "sort_into_cups": {
        "description": "Sort lego blocks by moving each block to its matching color cup position. Optionally filter by color.",
        "parameters": {
            "color": {
                "type": "string",
                "description": "Optional color to sort ('red', 'green', 'blue'). If omitted, sorts all colors."
            }
        }
    },
    "update_cups": {
        "description": "Reposition existing cups to match current CUP_LAYOUT settings WITHOUT rebuilding the /drop_poses action graph. Cup prims remain in place; only their xform + velocity are reset. If cups don't yet exist, falls back to delete+add (first-time creation path). Safe to call mid-simulation — /drop_poses keeps publishing throughout.",
        "parameters": {}
    },
    "match_real_world": {
        "description": "Reset CUP_LAYOUT to the real-world calibration values (CUP_LAYOUT_REAL_WORLD: cluster_offset_x, cluster_offset_y, gap) and reposition cups accordingly. This is the cup-setup branch of quick_start; call it from a reset script to return cups to the same poses they have on a fresh quick_start, regardless of any slider tweaks made since. Pass reposition=False to update CUP_LAYOUT in-place without moving existing cup prims.",
        "parameters": {
            "reposition": {"type": "boolean", "description": "if true (default), re-place cups via add_cups_from_ui so the values take immediate effect. False just updates CUP_LAYOUT.", "default": True}
        }
    },
    "randomize_cups": {
        "description": "Randomize cup layout by sampling CUP_LAYOUT params (mode/radius/angle/gap) and delegating placement to the existing _cup_positions_arc / _cup_positions_line generators. Each random sample is a coherent 3-cup layout (proper inter-cup spacing built in) — the function retries with fresh params until the resulting placement clears every lego footprint AND the robot's link AABBs (live-queried, so adaptive to the arm's current pose; call after grasp_home for the home-pose corridor exclusion). Defaults: radius ∈ [0.22, 0.30] m, angle ∈ [-50°, +50°], gap ∈ [0.005, 0.030] m, modes=('arc','line'). Yaw is anchored on the face-origin direction (ArUco markers face camera) with ±15° jitter. Cups teleported via the same Kit-command path as update_cups; /drop_poses wrappers auto-refreshed. After success, CUP_LAYOUT is left at the random params so subsequent 'Update' clicks reproduce the same layout.",
        "parameters": {
            "radius_min": {"type": "number", "description": "layout radius min (m)", "default": 0.22},
            "radius_max": {"type": "number", "description": "layout radius max (m)", "default": 0.30},
            "angle_min_deg": {"type": "number", "description": "layout heading min (deg, 0 = forward)", "default": -50.0},
            "angle_max_deg": {"type": "number", "description": "layout heading max (deg)", "default": 50.0},
            "gap_min": {"type": "number", "description": "cup-to-cup gap min (m)", "default": 0.005},
            "gap_max": {"type": "number", "description": "cup-to-cup gap max (m)", "default": 0.030},
            "modes": {"type": "array", "description": "list of layout modes to sample from. Default ['arc','line']", "required": False},
            "randomize_order": {"type": "boolean", "description": "shuffle color → slot mapping each sample (6 permutations for red/green/blue). False keeps canonical order", "default": True},
            "yaw_jitter_deg": {"type": "number", "description": "half-width of yaw jitter around face-origin (deg). 0 = always face origin exactly", "default": 15.0},
            "cup_lego_clearance": {"type": "number", "description": "extra margin (m) for cup-lego check (gripper needs room to approach lego — too small and pickup IK fails)", "default": 0.04},
            "cup_robot_clearance": {"type": "number", "description": "extra margin (m) for cup-robot-link check", "default": 0.015},
            "max_attempts": {"type": "integer", "description": "full-layout samples to try before failing", "default": 200},
            "seed": {"type": "integer", "description": "RNG seed for reproducibility (default: nondeterministic)", "required": False}
        }
    },
    "quick_start": {
        "description": "One-click setup: loads scene, robot, action graph, force publisher, wrist camera, objects, cups, publishers, and starts simulation.",
        "parameters": {}
    },
    "new_stage": {
        "description": "Clear the entire USD stage and free accumulated memory. Tears down all USD prims, Hydra render state, and PhysX allocations. Scene will be empty afterwards — call quick_start to rebuild.",
        "parameters": {}
    },
    "start_viewport_publisher": {
        "description": "Point the ACTIVE Isaac Sim viewport at the given camera prim and publish its frames to a ROS2 Image topic. Reuses the already-rendered viewport frame — zero extra render cost. Only one camera can be published this way at a time (switches active viewport camera). For multi-camera simultaneous publishing keep the action-graph-based setup instead.",
        "parameters": {
            "camera_prim_path": {
                "type": "string",
                "description": "USD path of the camera prim, e.g. /World/workspace_camera"
            },
            "topic_name": {
                "type": "string",
                "description": "ROS2 topic name (no leading slash), e.g. workspace_camera_rgb_sim"
            },
            "frame_id": {
                "type": "string",
                "description": "Optional ROS frame_id for the Image message header. Defaults to topic_name."
            }
        }
    },
    "stop_viewport_publisher": {
        "description": "Stop a viewport-publisher-based camera publisher by topic name.",
        "parameters": {
            "topic_name": {
                "type": "string",
                "description": "Topic name previously passed to start_viewport_publisher"
            }
        }
    },
    "list_viewport_publishers": {
        "description": "List active viewport-publisher camera topics with frame counts.",
        "parameters": {}
    },
    "start_recording": {
        "description": "Start real-time viewport video recording using ffmpeg. Records frames from the viewport render buffer without blocking physics. Returns immediately — call stop_recording to finalize the MP4.",
        "parameters": {
            "file_name": {
                "type": "string",
                "description": "Output filename (e.g. 'my_recording.mp4'). Saved under VIDEOS_DIR. Defaults to 'realtime_recording.mp4'."
            },
            "camera": {
                "type": "string",
                "description": "Camera prim path to record (switches active viewport to this camera). If omitted, uses whatever camera is already active."
            },
            "fps": {
                "type": "integer",
                "description": "Frames per second for the output video. Defaults to 30."
            }
        }
    },
    "stop_recording": {
        "description": "Stop the current real-time recording and finalize the MP4 video file. Returns the output file path and frame count.",
        "parameters": {}
    },
    "get_recording_status": {
        "description": "Check if viewport video recording is active. Returns status, frame count, elapsed time, output path, and FPS.",
        "parameters": {}
    },
    "get_contact_events": {
        "description": "Drain buffered PhysX contact events (subscribed via omni.physx contact-report subscription on cups + gripper/jaw/wrist). Returns a list of events: {t, type (CONTACT_FOUND/PERSIST/LOST), a0/a1 (actor paths), c0/c1 (collider paths), impulse [x,y,z], impulse_mag, position, n_contacts}. Events involving at least one watched prim are kept; ground-plane-only contacts are filtered. Used by motion_logger at 50 Hz to attribute cup displacement to specific robot-link pairs. Requires quick_start.",
        "parameters": {
            "drain": {"type": "boolean", "description": "If true (default), empty the buffer after reading. If false, return a snapshot of the tail and keep events."},
            "max_events": {"type": "integer", "description": "Max events to return in one call. Defaults to 256."}
        }
    },
}

# Handler method names for each tool (maps to self._cmd_<name> methods)
MCP_HANDLERS = {
    "play_scene": "_cmd_play_scene",
    "stop_scene": "_cmd_stop_scene",
    "sort_objects": "_cmd_sort_objects",
    "randomize_object_poses": "_cmd_randomize_object_poses",
    "randomize_single_object": "_cmd_randomize_single_object",
    "save_scene_state": "_cmd_save_scene_state",
    "restore_scene_state": "_cmd_restore_scene_state",
    "add_objects": "_cmd_add_objects",
    "delete_objects": "_cmd_delete_objects",
    "setup_pose_publisher": "_cmd_setup_pose_publisher",
    "sync_real_poses": "_cmd_sync_real_poses",
    "load_scene": "_cmd_load_scene",
    "load_robot": "_cmd_load_robot",
    "setup_action_graph": "_cmd_setup_action_graph",
    "setup_force_publisher": "_cmd_setup_force_publisher",
    "setup_bbox_publisher": "_cmd_setup_bbox_publisher",
    "setup_wrist_camera_action_graph": "_cmd_setup_wrist_camera_action_graph",
    "add_cups": "_cmd_add_cups",
    "publish_drop_poses": "_cmd_publish_drop_poses",
    "delete_cups": "_cmd_delete_cups",
    "sort_into_cups": "_cmd_sort_into_cups",
    "update_cups": "_cmd_update_cups",
    "match_real_world": "_cmd_match_real_world",
    "randomize_cups": "_cmd_randomize_cups",
    "quick_start": "_cmd_quick_start",
    "new_stage": "_cmd_new_stage",
    "start_viewport_publisher": "_cmd_start_viewport_publisher",
    "stop_viewport_publisher": "_cmd_stop_viewport_publisher",
    "list_viewport_publishers": "_cmd_list_viewport_publishers",
    "start_recording": "_cmd_start_recording",
    "stop_recording": "_cmd_stop_recording",
    "get_recording_status": "_cmd_get_recording_status",
    "get_contact_events": "_cmd_get_contact_events",
}
