pub mod coco_types;

use std::{collections::HashMap, fs::File, hash::Hash, io::BufReader, ops::Range, path::PathBuf};

use anyhow::Result;
use log::{debug, error, info, warn};

pub use coco_types::*;
#[derive(Debug)]
struct COCOIndices {
    image_id_to_image: HashMap<u32, Range<usize>>,
    annotation_id_to_annotation: HashMap<u128, Range<usize>>,
    category_id_to_category: HashMap<u32, Range<usize>>,
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

        let mut json_data: COCODetection = serde_json::from_reader(file_buffer)
            .inspect_err(|e| error!("Error deserializing json file: {e}"))?;

        // After serde has deserialized the json into COCODetection, we have two choices:
        // Create HashMaps with id as key and then indices to the original structs as value
        // This will result in multiple heap allocations per id and as we dont know how many
        // structs are there for a id, the vector resize will incur overhead as well
        //
        // We do know the total number of structs beforehand. The sum total of all the
        // hashmap's values' elements will be equal to that. We can utilize this to
        // allocate one big memory chunk at once and then create sub vectors from that memory
        // That will not only result in less allocations but the memory will be tightly packed
        //
        // I dont think this is possible in vanilla rust. For that reason, this is the alt sol:
        // sort the annotations and images in place (once they are read from json, user shouldnt be able to mutate them)
        // Store ranges for id [start_pos..end_pos(inclusive)]

        json_data.sort_images_inplace();
        json_data.sort_annots_inplace();
        json_data.sort_categories_inplace();

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
}

impl COCOIndices {
    fn create_hashmap<T>(iterator: impl ExactSizeIterator<Item = T>) -> HashMap<T::Id, Range<usize>>
    where
        T: COCOEntry,
        T::Id: Eq + Hash + Copy, // Eq because we are comparing, Hash because
                                 //it is being inserted in a hashmap as key,
                                 // Copy because it gets copied to hashmap as key
    {
        // Create a hashmap to store entry_id -> start..end
        let mut image_id_to_indices: HashMap<T::Id, Range<usize>> =
            HashMap::with_capacity(iterator.len());

        let mut current_image_id: Option<T::Id> = None;
        let mut current_image_id_start: Option<usize> = None;
        let mut current_image_id_end: Option<usize> = None;

        for (curr_index, coco_image) in iterator.enumerate() {
            if current_image_id.is_none() {
                current_image_id = Some(coco_image.get_id());
                current_image_id_start = Some(curr_index);
                current_image_id_end = Some(curr_index);
                continue;
            }

            if current_image_id.unwrap() == coco_image.get_id() {
                current_image_id_end = Some(current_image_id_end.unwrap() + 1);
            } else {
                image_id_to_indices.insert(
                    current_image_id.unwrap(),
                    current_image_id_start.unwrap()..current_image_id_end.unwrap() + 1,
                );

                current_image_id = Some(coco_image.get_id());
                current_image_id_start = Some(curr_index);
                current_image_id_end = Some(curr_index);
            }
        }

        image_id_to_indices.insert(
            current_image_id.unwrap(),
            current_image_id_start.unwrap()..current_image_id_end.unwrap() + 1,
        );
        image_id_to_indices
    }

    fn new(coco_detections: &COCODetection) -> Self {
        // We have our annotations and images sorted already according to their ids
        //
        let image_id_to_indices = COCOIndices::create_hashmap(coco_detections.iter_images());

        let annotation_id_to_indices =
            COCOIndices::create_hashmap(coco_detections.iter_annotations());

        let category_id_to_indices = COCOIndices::create_hashmap(coco_detections.iter_categories());

        Self {
            image_id_to_image: image_id_to_indices,
            annotation_id_to_annotation: annotation_id_to_indices,
            category_id_to_category: category_id_to_indices,
        }
    }
}
