pub mod coco_types;

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    hash::Hash,
    io::BufReader,
    ops::Range,
    path::PathBuf,
};

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
}

impl COCO {
    pub fn new(root_dir: PathBuf, annotation_json_path: PathBuf) -> Result<Self> {
        let file = File::open(annotation_json_path)
            .inspect_err(|e| error!("Error Opening Annotation file: {e}"))?;

        let file_buffer = BufReader::new(file);

        let json_data: COCODetection = serde_json::from_reader(file_buffer)
            .inspect_err(|e| error!("Error deserializing json file: {e}"))?;

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
        })
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
                Some(
                    filtered_annots
                        .into_iter()
                        .filter(|annot| category_ids.unwrap().contains(&annot.category_id))
                        .collect(),
                )
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
    use std::sync::OnceLock;

    // The shared, global instance initialized exactly once
    static COCO_INSTANCE: OnceLock<COCO> = OnceLock::new();

    fn get_shared_coco() -> &'static COCO {
        COCO_INSTANCE.get_or_init(|| {
            let ann_file = "./test_assets/stuff_val2017.json";
            let root_dir = ".";

            COCO::new(PathBuf::from(root_dir), PathBuf::from(ann_file))
                .expect("Failed to load global test COCO dataset")
        })
    }

    #[test]
    fn test_annotation_count_for_image() {
        let coco = get_shared_coco();
        let ann_ids_test_1 = coco.coco_indices.image_id_to_annotation_ids.get(&581781);

        let ann_ids_test_2 = coco.coco_indices.image_id_to_annotation_ids.get(&170278);

        let ann_ids_test_3 = coco.coco_indices.image_id_to_annotation_ids.get(&0);

        assert_eq!(ann_ids_test_1.unwrap().len(), 5);
        assert_eq!(ann_ids_test_2.unwrap().len(), 7);
        assert_eq!(ann_ids_test_3, None);
    }

    #[test]
    fn test_exact_annotation_ids() {
        let coco = get_shared_coco();
        let ann_ids_test_1 = coco
            .coco_indices
            .image_id_to_annotation_ids
            .get(&581781)
            .unwrap();

        let ann_ids_test_2 = coco
            .coco_indices
            .image_id_to_annotation_ids
            .get(&170278)
            .unwrap();

        let ann_ids_test_3 = coco.coco_indices.image_id_to_annotation_ids.get(&0);

        assert_eq!(ann_ids_test_1, &(20032796..20032801).collect::<Vec<u128>>());

        assert_eq!(ann_ids_test_2, &(20009250..20009257).collect::<Vec<u128>>());

        assert_eq!(ann_ids_test_3, None);
    }

    #[test]
    fn test_cat_ids_len() {
        // 103, 108, 131
        let coco = get_shared_coco();
        let cat_ids_test_1 = coco
            .coco_indices
            .category_id_to_image_ids
            .get(&103)
            .unwrap();

        let cat_ids_test_2 = coco
            .coco_indices
            .category_id_to_image_ids
            .get(&108)
            .unwrap();

        let ann_ids_test_3 = coco.coco_indices.category_id_to_image_ids.get(&0);

        assert_eq!(cat_ids_test_1.len(), 12);

        assert_eq!(cat_ids_test_2.len(), 20);

        assert_eq!(ann_ids_test_3, None);
    }

    #[test]
    fn test_exact_cat_ids() {
        // 103, 108, 131
        let coco = get_shared_coco();
        let cat_ids_test_1 = coco
            .coco_indices
            .category_id_to_image_ids
            .get(&103)
            .unwrap();

        let cat_ids_test_2 = coco
            .coco_indices
            .category_id_to_image_ids
            .get(&108)
            .unwrap();

        let ann_ids_test_3 = coco.coco_indices.category_id_to_image_ids.get(&0);

        assert_eq!(
            cat_ids_test_1,
            &[
                120584, 171757, 173302, 199771, 352582, 352684, 353970, 405195, 417249, 425221,
                483999, 491497
            ]
        );

        assert_eq!(
            cat_ids_test_2,
            &[
                22705, 45229, 54654, 146155, 175364, 190236, 192047, 205514, 222825, 229311,
                242934, 248400, 287714, 297353, 415741, 416343, 481386, 488673, 517056, 532901
            ]
        );

        assert_eq!(ann_ids_test_3, None);
    }
}
