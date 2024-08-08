# Required for relative imports when using this script
if __name__ == "__main__" and __package__ is None:
    __package__ = "infrastructure_tools.calibration"

    # Enable Excalibur import

    import os
    os.environ["ENABLE_CALIBRATION"] = "1"

import argparse
from copy import deepcopy
from pathlib import Path
from sys import exit, stdout
import yaml

import numpy as np

from .definitions import Correspondence, CalibrationData, CalibrationConfig, Hypothesis
from .routines import initial_scoring_and_clustering, estimate_multiple_hypotheses, hypothesis_refinement, merge_hypotheses


def calibrate_camera(calibration_data: CalibrationData, config: CalibrationConfig, force_merge: bool, disable_clustering: bool=False, use_single_hypothesis: bool=False, verbose: bool=False):
    hypotheses = estimate_multiple_hypotheses(calibration_data, config, verbose)

    filtered_hypotheses, hypothesis_scores = initial_scoring_and_clustering(calibration_data, hypotheses, config, disable_clustering, use_single_hypothesis, verbose)

    filtered_hypotheses_original = deepcopy(filtered_hypotheses)

    all_results = []
    all_scores = [] 

    for hypothesis_group in filtered_hypotheses:
        group_results = []
        group_scores = []
        for hypothesis_index in range(len(hypothesis_group) - 1, -1, -1):
            hypothesis_id = hypothesis_group[hypothesis_index]
            hypothesis = hypotheses[hypothesis_id]
            correspondence = calibration_data[hypothesis.data_id]
            solution, offsets_and_dists, used_timestamps, hypothesis_score = hypothesis_refinement(hypothesis, hypothesis_scores, correspondence, config, from_merged=False, verbose=verbose)
            if solution is not None:
                group_results.append((solution, offsets_and_dists, used_timestamps))
                group_scores.append(hypothesis_score)
        if len(group_results) > 0:
            all_results.append(group_results)
            all_scores.append(np.array(group_scores).mean())  
        del hypothesis_group[hypothesis_index]

    if verbose:
        print(f"Final scores: {all_scores}")

    exported_results = []
    
    for current_best_index in np.flip(np.argsort(np.array(all_scores))):
        exported_results += all_results[current_best_index]

        if verbose:
            print(f"Final candidates: {filtered_hypotheses[current_best_index]}")

        used_copy = False

        try:
            target_list = filtered_hypotheses[current_best_index]

            # Sometimes the tracks are not refinable, but still correct (linear dependencies, not enough excitation) => try to merge anyway
            if len(target_list) == 0:
                print("WARNING: Separate optimization failed!")
                target_list = filtered_hypotheses_original[current_best_index]
                used_copy = True

            merged_calib_track, timestamp_overlap = merge_hypotheses(calibration_data, [hypotheses[id_] for id_ in target_list], filter_inliers=True, verbose=verbose)
        except Exception:
            import traceback
            print(traceback.format_exc())
            print("Merge failed, falling back to single tracks")
            return exported_results

        all_hypotheses = [hypotheses[_i] for _i in filtered_hypotheses[current_best_index]] if not used_copy else [hypotheses[_i]
                                                                                                        for _i in
                                                                                                        filtered_hypotheses_original[
                                                                                                            current_best_index]]

        for hypothesis in all_hypotheses:
            # TODO: If inliers are not filtered during merging, currently all measurements from the merged tracks used, should depend on the filtering setting though
            hypothesis.inlier_ids = np.arange(merged_calib_track.gnss.shape[0], dtype=int)
            solution, offsets_and_dists, used_timestamps, _ = hypothesis_refinement(hypothesis, hypothesis_scores, merged_calib_track, config, from_merged=True, verbose=verbose)
            if solution is not None:
                exported_results.append((solution, offsets_and_dists, used_timestamps))
        
        if force_merge:
            # copy_index = None
            # for t_index, tup in enumerate(filtered_hypotheses_original):
            #     if filtered_hypotheses[current_best_index][0] in tup:
            #         copy_index = t_index
            #         break
            # assert copy_index is not None
            # target_list = filtered_hypotheses_original[copy_index]
            # merged_calib_track, timestamp_overlap = merge_hypotheses(calibration_data, [hypotheses[id_] for id_ in target_list], filter_inliers=True, verbose=verbose)
            pass

    return exported_results


def cli(args):

    if args.spu_id != -1 and args.cam_id != -1:
        camera_name = f"camera_spu_{args.spu_id}_{args.cam_id}"
    else:
        camera_name = args.camera_name

    config = CalibrationConfig.load(CalibrationConfig, args.config_file, camera_name=camera_name, platform_name=args.platform_name)

    calibration_data = CalibrationData.load_data(args.imu_file, args.det_file, config, args.time_offset)

    results = calibrate_camera(calibration_data, config, force_merge=args.force, disable_clustering=args.disable_clustering, use_single_hypothesis=args.use_single_hypothesis, verbose=args.verbose)

    if len(results) == 0:
        print("No valid solutions found!")
        exit(1)

    from ..utils.math import get_camera_position_4x4

    solutions = [res[0] for res in results]

    dists = np.array([np.array(res[1][0]).mean() for res in results])

    min_index = np.argmin(dists)

    for i, (solution, n, b) in enumerate(solutions):
        if i == min_index:
            print(f"Best candidate with a mean reprojection error of {dists[i]:.3f}m:")
        print("R:")
        print(solution[:3, :3])
        print("t:")
        print(solution[:3, 3])
        print("n:")
        print(n)
        print("b:")
        print(b)
        print("Orientation:")
        r1 = np.linalg.inv(solution) @ np.array([0, 0, 1, 1])
        r2 = np.linalg.inv(solution) @ np.array([0, -1, 1, 1])
        yaw = r2 - r1
        yaw = 180 / np.pi * np.arctan2(yaw[1], yaw[0])
        print(f"{yaw:.2f} deg")
        print("Position:")
        print(f"{get_camera_position_4x4(solution)}, mean error: {dists[i]:.3f}m")

        if i == min_index or args.verbose:
            print("Paste this into the camera config in the calibration section")
            print("="*50)
            calib_data = {}
            calib_data["K"] = config.k_matrix.tolist()
            calib_data["R"] = solution[:3, :3].tolist()
            calib_data["t"] = solution[:3, 3].tolist()
            calib_data["n"] = n.tolist()
            calib_data["b"] = b.tolist()
            yaml.dump(calib_data, stdout)
            print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate calibrator performance')
    parser.add_argument("-a", "--imu-file", type=Path,
                        help="IMU data path", required=True)
    parser.add_argument("-d", "--det-file", type=Path,
                        help="detection data path", required=True)
    # parser.add_argument("-r", "--radar-file", type=Path,
    #                     help="radar data path", required=False)
    parser.add_argument("-y", "--config-file", type=Path,
                        help="config data path", required=True)
    parser.add_argument("-s", "--spu-id", type=int, default=-1, help="SPU ID")
    parser.add_argument("-c", "--cam-id", type=int, default=-1, help="Camera ID")
    parser.add_argument("-f", "--force", type=bool,
                        default=False, help="Force track merging")
    parser.add_argument("-o", "--time-offset", type=int,
                        default=0, help="Time offset (37000000000 for TAI vehicle to GPS infrastructure)")
    parser.add_argument("-v", "--verbose",
                        help="Print debug information", action="store_true")
    parser.add_argument("--platform-name", type=str,
                        default="intersection_spu", help="Platform name", required=False)
    parser.add_argument("--camera-name", type=str,
                        help="Camera name as defined in platform_metadata", required=False)
    parser.add_argument("--use-single-hypothesis",
                        help="Don't discard single hypotheses", action="store_true")
    parser.add_argument("--disable-clustering",
                        help="Disable clustering", action="store_true")
    # parser.add_argument("-g", "--full_cam", type=bool,
    #                     default=False, help="Expect CAM yamls")
    # parser.add_argument("-t", "--type", type=str,
    #                     default="eval_real", help="Evaluation type", choices=["eval_real", "eval_fusion"],
    #                     required=False)
    # parser.add_argument("-i", "--input_file", type=str,
    #                     default="", help="Input file for evaluation")

    args = parser.parse_args()

    cli(args)
