use std::{collections::HashMap, prelude};

use log::{error, warn};
use memmap2::Advice::PopulateRead;

use crate::coco::{Annotation, COCO, COCOPredictions, Prediction};

use super::coco_types;

pub struct COCOEvalParams {
    iou_thresholds: Vec<f64>,
    recall_thresholds: Vec<f64>,
    max_detections: Vec<u8>,
    area_ranges: Vec<(f64, f64)>,
    use_categories: bool,
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
        }
    }
}

pub struct COCOEval<'a> {
    evaluation_parameters: COCOEvalParams,
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
    ious: &mut HashMap<(u32, u32), Vec<f32>>,
    // predictions: &Vec<&Prediction>,
    // gt_annotations: &Vec<&Annotation>,
    category_id: u32,
    image_id: u32,
    gt_bboxes: &Vec<&[f32; 4]>,
    pred_bboxes: &Vec<&[f32; 4]>,
) -> () {
    // if let Some(val) = ious.get_mut(&(image_id, category_id)) {
    //     val.reserve(gt_bboxes.len() * pred_bboxes.len())
    // }
    //
    ious.insert(
        (image_id, category_id),
        Vec::with_capacity(gt_bboxes.len() * pred_bboxes.len()),
    );

    for gt_bbox in gt_bboxes {
        for pred_bbox in pred_bboxes {
            // ious.insert((image_id, category_id), iou(*gt_bbox, *pred_bbox));
            ious.entry((image_id, category_id))
                .and_modify(|inner_vec| inner_vec.push(iou(*gt_bbox, *pred_bbox)));
            // ious.
        }
    }
}

impl<'a> COCOEval<'a> {
    #[inline(never)]
    pub fn perform_evaluation(&self) -> Option<()> {
        let gt_annotations = self.coco_dataset.get_all_annotations();
        let predictions = self.coco_dataset.get_all_results();
        if predictions.is_none() {
            error!("No results");
            return None;
        }

        let predictions = predictions.unwrap();

        // let image_ids_gt = self.coco_dataset.get_all_image_ids();
        // let category_ids_gt = self.coco_dataset.get_all_category_ids();

        let mut gt_annotations_eval_map: HashMap<(u32, u32), Vec<&[f32; 4]>> =
            HashMap::with_capacity(gt_annotations.len());

        let mut pred_eval_map: HashMap<(u32, u32), Vec<&[f32; 4]>> =
            HashMap::with_capacity(predictions.predictions.len());

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

        // dbg!(&pred_eval_map.keys());
        // dbg!(&gt_annotations_eval_map.keys());

        // // let predictions_vec = predictions.unwrap();
        // // let requested_max = usize::from(*self.evaluation_parameters.max_detections.last().unwrap());
        // // let end_index = predictions_vec.len().min(requested_max);
        // // let max_predictions: &[COCOPrediction] = &predictions_vec[0..end_index];
        // //
        // //

        let image_ids_gt = self.coco_dataset.get_all_image_ids();
        let category_ids_gt = self.coco_dataset.get_all_category_ids();

        let mut ious: HashMap<(u32, u32), Vec<f32>> =
            HashMap::with_capacity(image_ids_gt.len() * category_ids_gt.len());

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

        // dbg!(&ious);

        // // let gt_evaluation: HashMap<(u32, u32), Vec<&Annotation>> =
        // //     HashMap::with_capacity(image_ids_gt.len() * category_ids_gt.len());

        // // let detection_evaluation: HashMap<(u32, u32), &COCOPrediction> =
        // //     HashMap::with_capacity(predictions.unwrap().predictions.len());

        return Some(());
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
