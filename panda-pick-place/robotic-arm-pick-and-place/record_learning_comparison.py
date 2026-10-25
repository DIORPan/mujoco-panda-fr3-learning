"""Render paired evaluation actions to a synchronized, annotated MP4.

Rendering refreshes a separate MjData, so it cannot alter the live solver.
No neural policy or expert is run: replay is checked against saved states.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reach_grasp_task import ReachGraspTask


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--output", type=Path, default=Path("media/learning_comparison.mp4"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    evaluation = args.evaluation.resolve()
    output = args.output.resolve() if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"Choose a new output path: {output}")
    plan = json.loads((evaluation / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    if plan.get("schema") != "reach_grasp_paired_evaluation_v1" or not summary["completed_batch"]:
        raise ValueError("Expected a completed paired evaluation.")
    matches = [pair for pair in summary["pairs"] if pair["seed"] == args.seed]
    if len(matches) != 1 or not matches[0]["initial_states_match"]:
        raise ValueError("Choose exactly one evaluated seed with matched initial states.")
    pair = matches[0]
    for name, expected in plan["task_source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Execution source changed: {name}")
    os.chdir(root)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    tasks, traces, errors = {}, {}, {}
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    for name in ("bc", "aggregated"):
        with np.load(evaluation / pair[name]["trace_file"], allow_pickle=False) as saved:
            traces[name] = {key: saved[key].copy() for key in saved.files}
        task = tasks[name] = ReachGraspTask(model)
        initial_observation = task.reset(args.seed)
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, task.data, state, spec)
        np.testing.assert_allclose(state, traces[name]["initial_state"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(initial_observation, traces[name]["observations"][0], rtol=0, atol=1e-10)
        errors[name] = 0.
    width, height, fps = 640, 400, 25
    renderer = mujoco.Renderer(model, width=width, height=height)
    render_data = mujoco.MjData(model)
    render_state = np.empty(mujoco.mj_stateSize(model, spec))
    large, small = ImageFont.load_default(size=22), ImageFont.load_default(size=17)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), (1280, 512), fps=fps, codec="libx264",
                                        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=7,
                                        macro_block_size=16, ffmpeg_log_level="error",
                                        output_params=["-movflags", "+faststart"])
    writer.send(None)
    frame_count, final_canvas = 0, None

    def frame(time_s):
        canvas = Image.new("RGB", (1280, 512), "#101a29")
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 8), f"Panda reach + grasp | paired seed {args.seed} | t={time_s:.2f} s", fill="white", font=large)
        for index, name in enumerate(("bc", "aggregated")):
            left = index * width
            title = "BC baseline (frozen)" if name == "bc" else "Expert-label fine-tuning (frozen)"
            draw.text((left + 18, 39), title, fill="#f6be69" if name == "bc" else "#70ddbe", font=large)
            # All forward computation for rendering stays off the live task data.
            mujoco.mj_getState(model, tasks[name].data, render_state, spec)
            mujoco.mj_setState(model, render_data, render_state, spec)
            mujoco.mj_forward(model, render_data)
            renderer.update_scene(render_data, camera="demo_cam")
            canvas.paste(Image.fromarray(renderer.render().copy()), (left, 72))
            observation = tasks[name].observe()
            result = "PASS" if pair[name]["policy_success"] else "FAIL"
            text = f"lift={observation[3]*100:5.1f} cm | end-of-episode result: {result}"
            draw.text((left + 18, 481), text, fill="white", font=small)
        return canvas

    try:
        canvas = frame(0.)
        writer.send(np.asarray(canvas)); frame_count += 1
        for step in range(400):
            for name in ("bc", "aggregated"):
                observation = tasks[name].observe()
                error = float(np.abs(observation - traces[name]["observations"][step]).max())
                errors[name] = max(errors[name], error)
                np.testing.assert_allclose(observation, traces[name]["observations"][step], rtol=0, atol=1e-6)
                tasks[name].step(traces[name]["applied_actions"][step])
            if (step + 1) % 2 == 0:
                canvas = frame((step + 1) * .02)
                writer.send(np.asarray(canvas)); frame_count += 1
                final_canvas = canvas
        for name in ("bc", "aggregated"):
            np.testing.assert_allclose(tasks[name].observe(), traces[name]["final_observation"], rtol=0, atol=1e-6)
            if tasks[name].result()["grasp_success"] != pair[name]["policy_success"]:
                raise RuntimeError("Rendered replay changed the episode outcome.")
    finally:
        writer.close()
        renderer.close()
    poster = output.with_suffix(".png")
    final_canvas.save(poster)
    report = {
        "schema": "paired_action_replay_video_v1", "seed": args.seed,
        "selection": "illustrative evaluated seed; full aggregate results use all declared evaluation seeds",
        "evaluation_plan_sha256": digest(evaluation / "plan.json"),
        "evaluation_summary_sha256": digest(evaluation / "summary.json"),
        "fps": fps, "frames": frame_count, "simulated_duration_s": 8., "encoded_duration_s": frame_count / fps,
        "frame_size": [1280, 512], "rendering_uses_separate_data": True,
        "replay_max_observation_abs_error": errors, "outcomes_match_recorded_evaluation": True,
        "video_sha256": digest(output), "exporter_sha256": digest(Path(__file__)),
        "ffmpeg": imageio_ffmpeg.get_ffmpeg_version(),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved verified action replay: {output}")
    print(f"frames={frame_count}, fps={fps}, replay_errors={errors}")


if __name__ == "__main__":
    main()
