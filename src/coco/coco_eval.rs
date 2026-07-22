use std::collections::HashMap;

use anyhow::Result;
use log::{error, warn};
use ndarray::concatenate;

use crate::coco::COCO;

use ndarray::{Zip, prelude::*};

#[derive(Debug)]
pub struct COCOEvalParams {
    iou_thresholds: Vec<f32>,
    recall_thresholds: Vec<f64>,
    max_detections: Vec<u8>,
    area_ranges: Vec<(f64, f64)>,
    use_categories: bool,
    image_ids: Option<Vec<u32>>,
    category_ids: Option<Vec<u32>>,
}

impl COCOEvalParams {
    pub fn new() -> Self {
        Self {
            iou_thresholds: vec![0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            recall_thresholds: vec![
                0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13,
                0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
                0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41,
                0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55,
                0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83,
                0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
                0.98, 0.99, 1.0,
            ],
            max_detections: vec![1, 10, 100],
            area_ranges: vec![
                (0_f64.powi(2), 1e5_f64.powi(2)),
                (0_f64.powi(2), 32_f64.powi(2)),
                (32_f64.powi(2), 96_f64.powi(2)),
                (96_f64.powi(2), 1e5_f64.powi(2)),
            ],
            use_categories: true,
            image_ids: None,
            category_ids: None,
        }
    }

    pub fn set_image_ids(&mut self, image_ids: Vec<u32>) {
        self.image_ids = Some(image_ids);
    }
    pub fn set_category_ids(&mut self, category_ids: Vec<u32>) {
        self.category_ids = Some(category_ids);
    }
}

#[derive(Debug)]
pub struct COCOEval<'a> {
    pub evaluation_parameters: COCOEvalParams,
    coco_dataset: &'a COCO,
}

#[inline(never)]
pub fn iou(bbox_a: &[f32; 4], bbox_b: &[f32; 4]) -> f32 {
    // calculating intersection of two squares
    // square a square b
    // max_starting(a,b) + min_ending(a,b)
    // width will be ending - start
    // if it comes negative then it means that end is before start (no overlap)
    let max_starting_width = bbox_a[0].max(bbox_b[0]);
    let min_ending_width = (bbox_a[0] + bbox_a[2]).min(bbox_b[0] + bbox_b[2]);
    let intersection_width = (min_ending_width - max_starting_width).max(0.0);

    let max_starting_height = bbox_a[1].max(bbox_b[1]);
    let min_ending_height = (bbox_a[1] + bbox_a[3]).min(bbox_b[1] + bbox_b[3]);
    let intersection_height = (min_ending_height - max_starting_height).max(0.0);

    let intersection = intersection_width * intersection_height;
    let union = (bbox_a[2] * bbox_a[3]) + (bbox_b[2] * bbox_b[3]) - intersection;

    intersection / union
}
#[inline(never)]
pub fn populate_ious(
    ious: &mut HashMap<(u32, u32), Array3<f32>>,
    category_id: u32,
    image_id: u32,
    gt_bboxes: &Vec<&[f32; 4]>,
    pred_bboxes: &Vec<&[f32; 4]>,
) -> () {
    let mut vec: Vec<f32> = Vec::with_capacity(gt_bboxes.len() * pred_bboxes.len() * 2);

    for pred_bbox in pred_bboxes {
        for gt_bbox in gt_bboxes {
            vec.push(iou(*gt_bbox, *pred_bbox));
            vec.push(pred_bbox[2] * pred_bbox[3]);
        }
    }

    ious.insert(
        (image_id, category_id),
        Array3::from_shape_vec((pred_bboxes.len(), gt_bboxes.len(), 2), vec).unwrap(),
    );
}

pub fn cumsum_axis1(mask: Array2<bool>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros(mask.raw_dim());

    Zip::from(out.lanes_mut(Axis(1)))
        .and(mask.lanes(Axis(1)))
        .for_each(|mut out_row, mask_row| {
            let mut acc = 0.0;
            for (&b, val) in mask_row.iter().zip(out_row.iter_mut()) {
                acc += b as u8 as f64; // true -> 1.0, false -> 0.0
                *val = acc;
            }
        });

    out
}

impl<'a> COCOEval<'a> {
    #[inline(never)]
    fn get_matches(
        &self,
        calculated_ious: &Array3<f32>,
        area_range: &(f64, f64),
        max_detections: u8,
    ) -> (Array2<bool>, Array2<bool>) {
        // Get the actual number of rows (or axis length) in the array
        let dt_len = calculated_ious.shape()[0];

        // Clamp max_detections so it never exceeds the actual dimension size
        let end = (max_detections as usize).min(dt_len);

        let filtered_detections = calculated_ious.slice(s![..end, .., ..]);

        let mut gt_matches: Array2<bool> = Array2::<bool>::default((
            self.evaluation_parameters.iou_thresholds.len(),
            calculated_ious.shape()[1],
        ));
        let mut det_matches: Array2<bool> = Array2::<bool>::default((
            self.evaluation_parameters.iou_thresholds.len(),
            filtered_detections.shape()[0],
        ));

        for (iou_threshold_idx, iou_threshold) in
            self.evaluation_parameters.iou_thresholds.iter().enumerate()
        {
            let mut gt_matches_row = gt_matches.row_mut(iou_threshold_idx);
            let mut det_matches_row = det_matches.row_mut(iou_threshold_idx);

            for (det_idx, gt_metrics_matrix) in filtered_detections.outer_iter().enumerate() {
                let mut matched_gt_idx: Option<usize> = None;

                for (gt_idx, metrics) in gt_metrics_matrix.outer_iter().enumerate() {
                    let matched_iou = metrics[0];
                    let det_area = metrics[1];

                    // 1. Check IoU threshold
                    if matched_iou < *iou_threshold {
                        continue;
                    }

                    // 2. Check Area Range [min_area, max_area]
                    if (det_area as f64) < area_range.0 || (det_area as f64) > area_range.1 {
                        continue;
                    }

                    // 3. Skip if Ground Truth is already matched for this IoU threshold
                    if gt_matches_row[gt_idx] {
                        continue;
                    }

                    // 4. Skip if Detection has already been assigned to a GT
                    if matched_gt_idx.is_some() {
                        break;
                    }

                    matched_gt_idx = Some(gt_idx);
                }

                // Record match if one was found
                if let Some(gt_idx) = matched_gt_idx {
                    gt_matches_row[gt_idx] = true;
                    det_matches_row[det_idx] = true;
                }
            }
        }

        (gt_matches, det_matches)
    }

    #[inline(never)]
    pub fn perform_evaluation(&mut self) -> Result<Option<()>> {
        let gt_annotations = self.coco_dataset.get_all_annotations();
        let predictions = self.coco_dataset.get_all_results();
        if predictions.is_none() {
            error!("No results");
            return Ok(None);
        }

        let predictions = predictions.unwrap();

        let mut gt_annotations_eval_map: HashMap<(u32, u32), Vec<&[f32; 4]>> =
            HashMap::with_capacity(gt_annotations.len());

        let mut pred_eval_map: HashMap<(u32, u32), Vec<&[f32; 4]>> =
            HashMap::with_capacity(predictions.predictions.len());

        self.evaluation_parameters
            .set_image_ids(self.coco_dataset.get_all_image_ids());
        self.evaluation_parameters
            .set_category_ids(self.coco_dataset.get_all_category_ids());

        for gt_annotation in gt_annotations {
            gt_annotations_eval_map
                .entry((gt_annotation.image_id, gt_annotation.category_id))
                .and_modify(|inner_vec| inner_vec.push(&gt_annotation.bbox))
                .or_insert(vec![&gt_annotation.bbox]);
        }

        for prediction in &predictions.predictions {
            pred_eval_map
                .entry((prediction.image_id, prediction.category_id))
                .and_modify(|inner_vec| inner_vec.push(&prediction.bbox))
                .or_insert(vec![&prediction.bbox]);
        }

        let mut ious: HashMap<(u32, u32), Array3<f32>> =
            HashMap::with_capacity(gt_annotations_eval_map.keys().len());

        for (gt_image_id, gt_category_id) in gt_annotations_eval_map.keys() {
            let gt_bbox = gt_annotations_eval_map.get(&(*gt_image_id, *gt_category_id));
            let pred_bbox = pred_eval_map.get(&(*gt_image_id, *gt_category_id));

            if pred_bbox.is_some() && gt_bbox.is_some() {
                populate_ious(
                    &mut ious,
                    *gt_category_id,
                    *gt_image_id,
                    gt_bbox.unwrap(),
                    pred_bbox.unwrap(),
                );
            } else {
                warn!("({gt_image_id}, {gt_category_id}) pair doesnt exist in predictions");
            }
        }
        let all_categories = self.evaluation_parameters.category_ids.as_ref().unwrap();
        let area_ranges = &self.evaluation_parameters.area_ranges;
        let all_gt_ids = self.evaluation_parameters.image_ids.as_ref().unwrap();

        let mut accumulation: Vec<Option<(Array2<bool>, Array2<bool>)>> =
            Vec::with_capacity(all_categories.len() * area_ranges.len() * all_gt_ids.len());
        for category_id in all_categories {
            for area_range in area_ranges {
                for gt_image_id in all_gt_ids {
                    if let Some(val) = ious.get(&(*gt_image_id, *category_id)) {
                        accumulation.push(Some(self.get_matches(
                            val,
                            area_range,
                            *self.evaluation_parameters.max_detections.last().unwrap(),
                        )));
                    } else {
                        accumulation.push(None);
                    }
                }
            }
        }

        for category_idx in 0..all_categories.len() {
            let category_stride = category_idx * area_ranges.len() * all_gt_ids.len();
            for area_rng_idx in 0..area_ranges.len() {
                let area_range_stride = area_rng_idx * all_gt_ids.len();
                for max_det in &self.evaluation_parameters.max_detections {
                    // get all accumulations for these filters
                    let start_idx = category_stride + area_range_stride;
                    let end_idx = start_idx + all_gt_ids.len();
                    let all_corresponding_gts = &accumulation[start_idx..end_idx];
                    let filtered_corresponding_gts: Vec<&(Array2<bool>, Array2<bool>)> =
                        all_corresponding_gts
                            .iter()
                            .filter_map(|elem| elem.as_ref())
                            .collect();

                    let max_det = *max_det as usize;

                    // We map the iterator on-the-fly directly into a Vec of views.
                    let tp_views: Vec<_> = filtered_corresponding_gts
                        .iter()
                        .map(|gt| {
                            let max_end_idx = gt.1.shape()[1].min(max_det);
                            gt.1.slice(s![.., ..max_end_idx])
                        })
                        .collect();

                    let tps: Array2<bool> = concatenate(Axis(1), &tp_views)
                        .expect("Row counts must match across all arrays");

                    let fps: Array2<bool> = !&tps;

                    let tp_sum = cumsum_axis1(tps);
                    let fp_sum = cumsum_axis1(fps);

                    // if filtered_corresponding_gts.
                }
            }
        }

        return Ok(Some(()));
    }
    #[inline(never)]
    pub fn new(coco_dataset: &'a COCO) -> Self {
        Self {
            evaluation_parameters: COCOEvalParams::new(),
            coco_dataset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    #[test]
    fn test_perfect_overlap() {
        let box_a = [10.0, 10.0, 50.0, 50.0];
        let box_b = [10.0, 10.0, 50.0, 50.0];

        let result = iou(&box_a, &box_b);
        assert!(
            (result - 1.0).abs() < EPSILON,
            "Expected IoU to be 1.0, got {}",
            result
        );
    }

    #[test]
    fn test_no_overlap() {
        let box_a = [0.0, 0.0, 10.0, 10.0];
        let box_b = [20.0, 20.0, 10.0, 10.0];

        let result = iou(&box_a, &box_b);
        assert!(
            (result - 0.0).abs() < EPSILON,
            "Expected IoU to be 0.0, got {}",
            result
        );
    }

    #[test]
    fn test_partial_overlap() {
        // Two 10x10 boxes shifting by 5 units diagonally
        // Intersection area: 5 * 5 = 25
        // Union area: 100 + 100 - 25 = 175
        // IoU: 25 / 175 = 0.142857
        let box_a = [0.0, 0.0, 10.0, 10.0];
        let box_b = [5.0, 5.0, 10.0, 10.0];

        let expected = 25.0 / 175.0;
        let result = iou(&box_a, &box_b);
        assert!(
            (result - expected).abs() < EPSILON,
            "Expected IoU to be {}, got {}",
            expected,
            result
        );
    }

    #[test]
    fn test_nested_box() {
        // Box B is entirely inside Box A
        // Box A area = 10 * 10 = 100
        // Box B area = 5 * 5 = 25
        // Intersection = 25
        // Union = 100 + 25 - 25 = 100
        // IoU = 25 / 100 = 0.25
        let box_a = [0.0, 0.0, 10.0, 10.0];
        let box_b = [2.0, 2.0, 5.0, 5.0];

        let expected = 0.25;
        let result = iou(&box_a, &box_b);
        assert!(
            (result - expected).abs() < EPSILON,
            "Expected IoU to be {}, got {}",
            expected,
            result
        );
    }
}
