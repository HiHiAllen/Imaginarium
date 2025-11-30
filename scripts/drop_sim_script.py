from __future__ import annotations
import math
from typing import Dict, Iterable, List, Optional, Sequence
import bpy

# Default settings
ACTIVE_DEFAULTS: Dict[str, float | str] = {
    "shape": "CONVEX_HULL",   # 效果稳定
    "mass": 1,
    "friction": 100.0,
    "bounciness": 0.0,
    "margin": 0.001,
    "linear_damping": 0.5,
    "angular_damping": 0.5,
}

PASSIVE_DEFAULTS: Dict[str, float | str] = {
    "shape": "MESH",    # 但这个要用MESH
    "friction": 100.0,
    "bounciness": 0.0,
    "margin": 0.001,
}

WORLD_DEFAULTS: Dict[str, float | int | bool] = {
    "frame_start": 1,
    "substeps": 10,
    "solver_iterations": 10,
    "split_impulse": True,
}


def _ensure_object_mode() -> None:
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")


def _iter_mesh_objects(objects: Optional[Iterable[bpy.types.Object]]) -> List[bpy.types.Object]:
    result: List[bpy.types.Object] = []
    if not objects:
        return result
    for obj in objects:
        if obj and obj.type == "MESH":
            result.append(obj)
    return result


def _select_only(objects: Sequence[bpy.types.Object]) -> None:
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in objects:
        obj.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]


def _ensure_rigidbody(obj: bpy.types.Object, body_type: str) -> bpy.types.RigidBodyObject:
    if not obj.rigid_body:
        _select_only([obj])
        bpy.ops.rigidbody.object_add(type=body_type)
    obj.rigid_body.type = body_type
    return obj.rigid_body


def _configure_active(obj: bpy.types.Object, settings: Dict[str, float | str]) -> None:
    rb = _ensure_rigidbody(obj, "ACTIVE")
    rb.mass = settings["mass"]
    rb.friction = settings["friction"]
    rb.restitution = settings["bounciness"]
    rb.collision_margin = settings["margin"]
    rb.use_margin = True
    rb.collision_shape = settings["shape"]
    rb.linear_damping = settings["linear_damping"]
    rb.angular_damping = settings["angular_damping"]
    rb.mesh_source = "FINAL"


def _configure_passive(obj: bpy.types.Object, settings: Dict[str, float | str]) -> None:
    rb = _ensure_rigidbody(obj, "PASSIVE")
    rb.friction = settings["friction"]
    rb.restitution = settings["bounciness"]
    rb.collision_margin = settings["margin"]
    rb.use_margin = True
    rb.collision_shape = settings["shape"]
    rb.kinematic = True
    rb.mesh_source = "FINAL"


def _ensure_world(scene: bpy.types.Scene, frame_end: int, world_settings: Dict[str, float | int | bool]) -> bpy.types.RigidBodyWorld:
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    rb_world = scene.rigidbody_world
    rb_world.enabled = True
    rb_world.use_split_impulse = bool(world_settings["split_impulse"])
    rb_world.substeps_per_frame = int(world_settings["substeps"])
    rb_world.solver_iterations = int(world_settings["solver_iterations"])

    if rb_world.point_cache:
        rb_world.point_cache.frame_start = int(world_settings["frame_start"])
        rb_world.point_cache.frame_end = frame_end
        rb_world.point_cache.frame_step = 1
        rb_world.point_cache.index = 0

    scene.frame_start = int(world_settings["frame_start"])
    scene.frame_end = frame_end
    scene.frame_set(scene.frame_start)
    return rb_world


def _simulate(scene: bpy.types.Scene, frame_end: int) -> None:
    import sys
    total_frames = frame_end - scene.frame_start + 1
    for i, frame in enumerate(range(scene.frame_start, frame_end + 1)):
        scene.frame_set(frame)
        # 每10帧或最后一帧输出进度
        if (i + 1) % 10 == 0 or frame == frame_end:
            progress = ((i + 1) / total_frames) * 100
            print(f"[PhysicsSimulation] 仿真进度: {i+1}/{total_frames} 帧 ({progress:.1f}%)", flush=True)
            sys.stdout.flush()


def _apply_and_cleanup(objects: Sequence[bpy.types.Object]) -> None:
    if not objects:
        return
    _select_only(objects)
    bpy.ops.object.visual_transform_apply()
    bpy.ops.rigidbody.objects_remove()
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)


def run_drop_simulation(
    objects: Iterable[bpy.types.Object],
    colliders: Optional[Iterable[bpy.types.Object]] = None,
    duration: float = 1.0,
    scene: Optional[bpy.types.Scene] = None,
    active_settings: Optional[Dict[str, float | str]] = None,
    passive_settings: Optional[Dict[str, float | str]] = None,
    world_settings: Optional[Dict[str, float | int | bool]] = None,
) -> bool:
    """
    Simulate a rigidbody drop for the provided objects and bake the result in-place.

    Args:
        objects: Iterable of mesh objects to treat as active rigidbodies.
        colliders: Optional iterable of mesh objects that should become passive colliders.
        duration: Simulation duration in seconds (default 1 seconds).
        scene: Optional scene override. Defaults to bpy.context.scene.
        active_settings/passive_settings/world_settings:
            Dictionaries overriding the defaults declared in this file.

    Returns:
        True when the simulation finishes successfully, False otherwise.
    """

    scene = scene or bpy.context.scene
    if not scene:
        return False

    _ensure_object_mode()

    active_objs = _iter_mesh_objects(objects)
    passive_objs = _iter_mesh_objects(colliders)

    if not active_objs:
        return False

    active_cfg = {**ACTIVE_DEFAULTS, **(active_settings or {})}
    passive_cfg = {**PASSIVE_DEFAULTS, **(passive_settings or {})}
    world_cfg = {**WORLD_DEFAULTS, **(world_settings or {})}

    fps = scene.render.fps / scene.render.fps_base
    total_frames = max(1, math.ceil(duration * fps))
    frame_end = world_cfg["frame_start"] + total_frames

    _ensure_world(scene, frame_end, world_cfg)

    collider_previous: Dict[bpy.types.Object, Optional[str]] = {}
    for collider in passive_objs:
        prev_type = collider.rigid_body.type if collider.rigid_body else None
        collider_previous[collider] = prev_type
        _configure_passive(collider, passive_cfg)

    for obj in active_objs:
        _configure_active(obj, active_cfg)

    _simulate(scene, frame_end)
    _apply_and_cleanup(active_objs)

    for collider in passive_objs:
        prev_type = collider_previous.get(collider)
        if prev_type is None:
            _select_only([collider])
            bpy.ops.rigidbody.objects_remove()
        else:
            if collider.rigid_body:
                collider.rigid_body.type = prev_type

    return True


def auto_run(duration: float = 3.0) -> bool:
    """Drop currently selected mesh objects against all other mesh objects."""
    scene = bpy.context.scene
    if not scene:
        print("[PhysicsDropScript] 无法取得当前场景。")
        return False

    selected_meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not selected_meshes:
        print("[PhysicsDropScript] 请先选择至少一个网格物体。")
        return False

    colliders = [obj for obj in scene.objects if obj.type == "MESH" and obj not in selected_meshes]
    success = run_drop_simulation(selected_meshes, colliders, duration=duration, scene=scene)

    if success:
        print(f"[PhysicsDropScript] 已对 {len(selected_meshes)} 个物体执行 {duration:.2f}s 下落仿真。")
    else:
        print("[PhysicsDropScript] 仿真失败，请检查场景设置。")
    return success


__all__ = ["run_drop_simulation", "auto_run"]


if __name__ == "__main__":
    auto_run(duration=3.0)

