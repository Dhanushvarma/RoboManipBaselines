# Misc

## Data utilities
The file format pointed to by `<rmb_file>` can be either RmbData-Compact (`.rmb`) or RmbData-SingleHDF5 (`.hdf5`).

### Visualize demonstration data
Visualize the demonstration data by plotting it.

```console
$ python ./VisualizeData.py <rmb_file>
```

### Switch internal format of demonstration data
Switch RMB format file between RmbData-Compact (`.rmb`) and RmbData-SingleHDF5 (`.hdf5`). The format is automatically determined from the file extension.

```console
$ python ./SwitchRmbDataFormat.py <rmb_file_in> <rmb_file_out>
```

### Compare demonstration data
Compare the contents of the two RMB format files to see if they match.
Note that when converting images to mp4 files, lossy compression is applied to color images and quantization is applied to depth images, so RmbData-Compact contains some errors.

```console
$ python ./CompareRmbData.py <rmb_file1> <rmb_file2>
```

### Refine demonstration data
Update the task description attribute in RMB format files. It accepts a path to a file or directory and automatically searches for relevant files. If the task description attribute exists and `--overwrite` is not specified, the value is not changed.

```console
$ python ./RefineRmbData.py <rmb_file> --task_desc "<new_description>" [--overwrite]
```

### Replay demonstration data verbatim
Replay recorded demonstrations through `RealUR5eEnvBase` (ur_rtde servoJ) and report
joint tracking error. This is the control baseline: the demos
were recorded through this same env, so anything wrong here is the robot or the
controller. Reports the recording's *own* tracking error alongside this run's, so a
replay can be judged against how well the arm tracked when the data was captured.
Requires the robot: the env opens its RTDE connection on construction.

```console
$ python ./ReplayRealUR5eDemo.py <rmb_file> --config ../envs/configs/RealUR5eDemoEnv.yaml
```

### Replay demonstration data through the B-spline stack
Replay a recorded demonstration at three levels of the B-spline policy stack, to
separate a lossy spline representation from a badly trained policy. Each mode adds
exactly one component: `raw` is the controller alone, `spline` adds the whole-episode
fit, and `segments` adds chunking, segment alignment and the high-rate sampler (a
perfect-prediction oracle rollout). `bin/Rollout.py` adds the trained network on top.
Prints a `demo -> fitted -> commanded -> measured` error decomposition, per joint in
degrees and the gripper in counts. Use `--dry_run` to inspect it with no hardware.

```console
$ python ./ReplayBsplineDemo.py <rmb_file> --config ../envs/configs/RealUR5eBSplineDemoEnv.yaml --mode segments
```

### Plot replay error vs speedup
Plot joint error against speedup from the CSVs written by the two replay scripts'
`--log` option. Writes six PNGs: maximum and RMS joint error, each for the fit,
execute and track stages, with one line per replay mode. Rows from both scripts share
one schema, so the `ur_rtde` baseline appears on the same axes as the B-spline modes.
Gripper columns are logged but not plotted.

```console
$ python ./ReplayBsplineDemo.py <rmb_file> --config ../envs/configs/RealUR5eBSplineDemoEnv.yaml \
    --mode segments --speedup 2.0 --log ./replay_errors.csv
$ python ./PlotReplayErrors.py ./replay_errors.csv --output_dir ./replay_error_plots
```

## Visualization utilities
### Plot B-spline fit quality
Fit demonstration data with the same settings training uses and plot the result. Fully
offline: no robot, no policy, no checkpoint. Writes one PNG per episode (demo vs fitted
per joint, reconstruction error against the tolerance, gripper, knot density) plus a
dataset summary (compression, per-joint error, segment spans, knots vs length). Saves
by default rather than opening windows, since forwarding figures over `ssh -X` is very
slow; pass `--show` for an interactive window.

```console
$ python ./PlotBsplineFit.py <rmb_file> --output_dir ./bspline_fit_plots
```

### Visualize camera images
Display the web camera image for recording the experiments.
```console
$ python ./DisplayCameraImage.py --camera_name Webcam --resize_width 800 --win_xy 1000 400
```

Display the cropped camera image. This is useful for image cropping policies such as SARNN.
```console
$ python ./DisplayCameraImage.py --camera_name RealSense --crop_size 280 280
```

A camera can also be specified by `--camera_id` instead of `--camera_name`.

## Video utilities
### Tile rollout videos
The input is a video consisting of a sequence of multiple rollouts, and the output is a tiled video of each rollout.
```console
$ python ./TileRolloutVideos.py <input_video_path> --output_file_name <output_video_path> --task_success_list 1 0 1 0 1 1 --column_num 3
```
The options `--output_file_name`, `--task_success_list`, and `--column_num` can be omitted.

By default, the video separation times are automatically determined by detecting restarts of windows in the video, but can be specified explicitly by adding the `--task_period_list` option as follows:
```console
--task_period_list 00:00.00-00:11.00 00:14.00-00:27.50 00:30.20-00:42.50 00:45.70-00:58.50 01:01.70-01:13.50 01:16.70-01:28.00
```
