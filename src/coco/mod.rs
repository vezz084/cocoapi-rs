pub mod coco_eval;
pub mod coco_types;

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    hash::Hash,
    io::BufReader,
    io::Read,
    ops::Range,
    path::PathBuf,
};

use memmap2::Mmap;

use anyhow::Result;
use log::{debug, error, info, warn};

pub use coco_types::*;
#[derive(Debug)]
struct COCOIndices {
    image_id_to_image: HashMap<u32, usize>,
    annotation_id_to_annotation: HashMap<u128, usize>,
    category_id_to_category: HashMap<u32, usize>,
    image_id_to_annotation_ids: HashMap<u32, Vec<u128>>,
    category_id_to_image_ids: HashMap<u32, Vec<u32>>,
    category_id_to_annotation_ids: HashMap<u32, Vec<u128>>,
}

#[derive(Debug)]
pub struct COCO {
    coco_dataset: COCODetection,
    root_dir: PathBuf,
    coco_indices: COCOIndices,
    coco_results: Option<COCOPredictions>,
}

impl COCO {
    pub fn new(root_dir: PathBuf, annotation_json_path: PathBuf) -> Result<Self> {
        let file = File::open(annotation_json_path)
            .inspect_err(|e| error!("Error Opening Annotation file: {e}"))?;

        // Memory map the file. (Almost a 2x speed up)
        // This takes near-zero RAM and happens instantly.
        let mmap = unsafe {
            Mmap::map(&file).inspect_err(|e| error!("Error Mapping The File to Memory: {e}"))?
        };
        // The alternative is loading the whole file in memory
        // let mut buffer = Vec::with_capacity(file.metadata()?.len() as usize);
        // file.read_to_end(&mut buffer)?;
        // But this consumes a lot of memory
        //
        // Must communicate with the user to make sure the file does not change while it is being read
        // OS level locks maybe?

        let json_data: COCODetection = serde_json::from_slice(&mmap)
            .inspect_err(|e| error!("Error Deserializing The File: {e}"))?;

        info!("Total Images: {}", json_data.iter_images().len());
        info!("Total Annotations: {}", json_data.iter_annotations().len());
        info!("Total Categories: {}", json_data.iter_categories().len());

        info!("Creating indices...");

        let indices = COCOIndices::new(&json_data);

        info!("Indices created!");

        Ok(COCO {
            coco_dataset: json_data,
            root_dir: root_dir,
            coco_indices: indices,
            coco_results: None,
        })
    }

    pub fn get_all_annotations(&self) -> Vec<&Annotation> {
        self.coco_dataset.iter_annotations().collect()
    }

    pub fn get_all_category_ids(&self) -> Vec<u32> {
        self.coco_dataset
            .iter_categories()
            .map(|coco_category| coco_category.id)
            .collect()
    }

    pub fn get_all_image_ids(&self) -> Vec<u32> {
        self.coco_dataset
            .iter_images()
            .map(|coco_image| coco_image.id)
            .collect()
    }

    pub fn get_all_results(&self) -> Option<&COCOPredictions> {
        if self.coco_results.is_none() {
            return None;
        }
        Some(self.coco_results.as_ref().unwrap())
    }

    pub fn get_images_from_ids(&self, ids: &[u32]) -> Vec<Option<&COCOImage>> {
        let mut result = Vec::with_capacity(ids.len());

        for image_id in ids {
            result.push(match self.coco_indices.image_id_to_image.get(image_id) {
                None => None,
                Some(index) => self.coco_dataset.get_image_at_index(*index),
            });
        }

        result
    }

    pub fn get_annotations_from_ids(&self, ids: &[u128]) -> Vec<Option<&Annotation>> {
        let mut result = Vec::with_capacity(ids.len());

        for annotation_id in ids {
            result.push(
                match self
                    .coco_indices
                    .annotation_id_to_annotation
                    .get(annotation_id)
                {
                    None => None,
                    Some(index) => self.coco_dataset.get_annotation_at_index(*index),
                },
            );
        }

        result
    }

    pub fn get_categories_from_ids(&self, ids: &[u32]) -> Vec<Option<&COCOCategory>> {
        let mut result = Vec::with_capacity(ids.len());

        for category_id in ids {
            result.push(
                match self.coco_indices.category_id_to_category.get(category_id) {
                    None => None,
                    Some(index) => self.coco_dataset.get_category_at_index(*index),
                },
            );
        }

        result
    }

    pub fn get_possible_annotations_from_image_ids(&self, ids: &[u32]) -> Option<Vec<&Annotation>> {
        if ids.is_empty() {
            return None;
        }

        let mut result = Vec::with_capacity(ids.len());

        for image_id in ids {
            if let Some(annot_ids) = self.coco_indices.image_id_to_annotation_ids.get(image_id) {
                result.extend(
                    self.get_annotations_from_ids(annot_ids)
                        .into_iter()
                        .flatten(),
                );
            }
        }

        if result.is_empty() {
            return None;
        }

        Some(result)
    }

    pub fn get_possible_annotations_from_category_ids(
        &self,
        ids: &[u32],
    ) -> Option<Vec<&Annotation>> {
        if ids.is_empty() {
            return None;
        }

        let mut result = Vec::with_capacity(ids.len());

        for category_id in ids {
            if let Some(annot_ids) = self
                .coco_indices
                .category_id_to_annotation_ids
                .get(category_id)
            {
                result.extend(
                    self.get_annotations_from_ids(annot_ids)
                        .into_iter()
                        .flatten(),
                );
            }
        }

        if result.is_empty() {
            return None;
        }

        Some(result)
    }

    pub fn get_possible_annotations_from_filters(
        &self,
        image_ids: Option<&[u32]>,
        category_ids: Option<&[u32]>,
    ) -> Option<Vec<&Annotation>> {
        if image_ids.is_none() && category_ids.is_none() {
            return None;
        }

        if image_ids.is_none() {
            // get only by categories
            self.get_possible_annotations_from_category_ids(category_ids.unwrap())
        } else if category_ids.is_none() {
            self.get_possible_annotations_from_image_ids(image_ids.unwrap())
        } else {
            let filtered_by_image_ids =
                self.get_possible_annotations_from_image_ids(image_ids.unwrap());
            if let Some(filtered_annots) = filtered_by_image_ids {
                let filtered: Vec<_> = filtered_annots
                    .into_iter()
                    .filter(|annot| category_ids.unwrap().contains(&annot.category_id))
                    .collect();

                if filtered.is_empty() {
                    None
                } else {
                    Some(filtered)
                }
            } else {
                return None;
            }
        }
    }

    pub fn get_possible_images_from_filters(
        &self,
        image_ids: Option<&[u32]>,
        category_ids: Option<&[u32]>,
    ) -> Option<Vec<&COCOImage>> {
        if image_ids.is_none() && category_ids.is_none() {
            return None;
        }

        let mut valid_ids: Vec<u32> = Vec::with_capacity(image_ids.unwrap_or(&[]).len());

        if image_ids.is_none() {
            // only filter by categories
            for category_id in category_ids.unwrap() {
                if let Some(image_id) = self.coco_indices.category_id_to_image_ids.get(category_id)
                {
                    valid_ids.extend(image_id.iter())
                }
            }
        } else {
            let filtered_annots = self.get_possible_annotations_from_image_ids(image_ids.unwrap());

            if category_ids.is_none() {
                if let Some(filtered_annots) = filtered_annots {
                    valid_ids.extend(filtered_annots.iter().map(|annot| annot.image_id));
                }
            } else {
                if let Some(filtered_annots) = filtered_annots {
                    for annot in filtered_annots {
                        if category_ids.unwrap().contains(&annot.category_id) {
                            valid_ids.push(annot.image_id);
                        }
                    }
                }
            }
        }

        if valid_ids.is_empty() {
            return None;
        }

        valid_ids.sort();
        valid_ids.dedup();

        let temp_vec: Vec<u32> = valid_ids.into_iter().collect();
        let ans = self.get_images_from_ids(&temp_vec);

        Some(ans.into_iter().flatten().collect())
    }

    pub fn load_coco_predictions(&mut self, prediction_file_path: PathBuf) -> Result<()> {
        let file = File::open(prediction_file_path)
            .inspect_err(|e| error!("Error opening prediction file: {e}"))?;
        let file_buffer = BufReader::new(file);

        let predictions: Vec<Prediction> = serde_json::from_reader(file_buffer)
            .inspect_err(|e| error!("Error deserializing predictions json file: {e}"))?;

        // 2. Wrap it into your COCOPredictions struct
        let mut coco_predictions = COCOPredictions { predictions };

        coco_predictions.sort_predictions_by_score_htl();

        self.coco_results = Some(coco_predictions);

        Ok(())
    }
}

impl COCOIndices {
    fn create_index<T>(iterator: impl ExactSizeIterator<Item = T>) -> HashMap<T::Id, usize>
    where
        T: COCOEntry,
        T::Id: Eq + Hash + Copy, // Eq because we are comparing, Hash because
                                 //it is being inserted in a hashmap as key,
                                 // Copy because it gets copied to hashmap as key
    {
        // Create a hashmap to store entry_id -> start..end
        let mut entry_id_to_index: HashMap<T::Id, usize> = HashMap::with_capacity(iterator.len());

        for (curr_index, coco_image) in iterator.enumerate() {
            entry_id_to_index.insert(coco_image.get_id(), curr_index);
        }

        entry_id_to_index
    }

    fn new(coco_detections: &COCODetection) -> COCOIndices {
        // We have our annotations sorted already according to their ids
        let image_id_to_image = COCOIndices::create_index(coco_detections.iter_images());
        let category_id_to_category = COCOIndices::create_index(coco_detections.iter_categories());
        let annotation_id_to_annotation =
            COCOIndices::create_index(coco_detections.iter_annotations());

        let mut image_id_to_annotation_ids: HashMap<u32, Vec<u128>> =
            HashMap::with_capacity(coco_detections.iter_images().len());
        let mut category_id_to_image_ids: HashMap<u32, Vec<u32>> =
            HashMap::with_capacity(coco_detections.iter_categories().len());
        let mut category_id_to_annotation_ids: HashMap<u32, Vec<u128>> =
            HashMap::with_capacity(coco_detections.iter_categories().len());

        for coco_annotation in coco_detections.iter_annotations() {
            image_id_to_annotation_ids
                .entry(coco_annotation.image_id)
                .or_default()
                .push(coco_annotation.id);
            category_id_to_annotation_ids
                .entry(coco_annotation.category_id)
                .or_default()
                .push(coco_annotation.id);

            category_id_to_image_ids
                .entry(coco_annotation.category_id)
                .or_default()
                .push(coco_annotation.image_id);
        }

        Self {
            image_id_to_image,
            annotation_id_to_annotation,
            category_id_to_category,
            image_id_to_annotation_ids,
            category_id_to_image_ids,
            category_id_to_annotation_ids,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    static COCO_INSTANCE: OnceLock<COCO> = OnceLock::new();

    fn get_shared_coco() -> &'static COCO {
        COCO_INSTANCE.get_or_init(|| {
            let ann_file = "./test_assets/stuff_val2017.json";
            let root_dir = ".";

            COCO::new(PathBuf::from(root_dir), PathBuf::from(ann_file))
                .expect("Failed to load global test COCO dataset")
        })
    }

    // --- IMAGE TO ANNOTATION ID MAPPING ---

    #[test]
    fn test_annotation_count_for_image_581781() {
        let coco = get_shared_coco();
        let ids = coco.coco_indices.image_id_to_annotation_ids.get(&581781);

        assert_eq!(
            ids.map(|v| v.len()),
            Some(5),
            "Image ID 581781 should have exactly 5 annotations"
        );
    }

    #[test]
    fn test_annotation_count_for_image_170278() {
        let coco = get_shared_coco();
        let ids = coco.coco_indices.image_id_to_annotation_ids.get(&170278);

        assert_eq!(
            ids.map(|v| v.len()),
            Some(7),
            "Image ID 170278 should have exactly 7 annotations"
        );
    }

    #[test]
    fn test_annotation_count_for_missing_image_0() {
        let coco = get_shared_coco();
        let ids = coco.coco_indices.image_id_to_annotation_ids.get(&0);

        assert!(
            ids.is_none(),
            "Non-existent Image ID 0 should return None for annotation mapping"
        );
    }

    #[test]
    fn test_exact_annotation_ids_for_image_581781() {
        let coco = get_shared_coco();
        let ids = coco.coco_indices.image_id_to_annotation_ids.get(&581781);

        assert_eq!(
            ids.map(|v| v.as_slice()),
            Some([20032796, 20032797, 20032798, 20032799, 20032800].as_slice()),
            "Mismatch in expected annotation ID sequence for Image ID 581781"
        );
    }

    #[test]
    fn test_exact_annotation_ids_for_image_170278() {
        let coco = get_shared_coco();
        let ids = coco.coco_indices.image_id_to_annotation_ids.get(&170278);

        assert_eq!(
            ids.map(|v| v.as_slice()),
            Some(
                [
                    20009250, 20009251, 20009252, 20009253, 20009254, 20009255, 20009256
                ]
                .as_slice()
            ),
            "Mismatch in expected annotation ID sequence for Image ID 170278"
        );
    }

    // --- CATEGORY TO IMAGE ID MAPPING ---

    #[test]
    fn test_cat_ids_len_for_category_103() {
        let coco = get_shared_coco();
        let ids = coco.coco_indices.category_id_to_image_ids.get(&103);

        assert_eq!(
            ids.map(|v| v.len()),
            Some(12),
            "Category ID 103 should map to exactly 12 images"
        );
    }

    #[test]
    fn test_cat_ids_len_for_category_108() {
        let coco = get_shared_coco();
        let ids = coco.coco_indices.category_id_to_image_ids.get(&108);

        assert_eq!(
            ids.map(|v| v.len()),
            Some(20),
            "Category ID 108 should map to exactly 20 images"
        );
    }

    #[test]
    fn test_cat_ids_for_missing_category_0() {
        let coco = get_shared_coco();
        let ids = coco.coco_indices.category_id_to_image_ids.get(&0);

        assert!(
            ids.is_none(),
            "Non-existent Category ID 0 should return None for image mapping"
        );
    }

    #[test]
    fn test_exact_image_ids_for_category_103() {
        let coco = get_shared_coco();
        let ids = coco.coco_indices.category_id_to_image_ids.get(&103);

        assert_eq!(
            ids.map(|v| v.as_slice()),
            Some(
                [
                    120584, 171757, 173302, 199771, 352582, 352684, 353970, 405195, 417249, 425221,
                    483999, 491497
                ]
                .as_slice()
            ),
            "Mismatch in expected image IDs for Category ID 103"
        );
    }

    #[test]
    fn test_exact_image_ids_for_category_108() {
        let coco = get_shared_coco();
        let ids = coco.coco_indices.category_id_to_image_ids.get(&108);

        assert_eq!(
            ids.map(|v| v.as_slice()),
            Some(
                [
                    22705, 45229, 54654, 146155, 175364, 190236, 192047, 205514, 222825, 229311,
                    242934, 248400, 287714, 297353, 415741, 416343, 481386, 488673, 517056, 532901
                ]
                .as_slice()
            ),
            "Mismatch in expected image IDs for Category ID 108"
        );
    }

    // --- STRUCTURAL INTEGRITY / INVARIANT TEST ---

    #[test]
    fn test_index_integrity_invariants() {
        let coco = get_shared_coco();
        let indices = &coco.coco_indices;

        // Verify that every image referenced by a category actually contains annotations
        for (cat_id, image_ids) in &indices.category_id_to_image_ids {
            for image_id in image_ids {
                let annotations = indices.image_id_to_annotation_ids.get(image_id);
                assert!(
                    annotations.is_some(),
                    "Integrity failure: Category {} references Image {}, but that image has no indexed annotations",
                    cat_id,
                    image_id
                );
            }
        }
    }
}
