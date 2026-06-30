pub mod coco_types;

use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Error},
    ops::Range,
    path::{Path, PathBuf},
};

pub use coco_types::*;
#[derive(Debug)]
struct COCOIndices {
    // image_id_to_filename: HashMap<&'a u32, PathBuf>,
    image_id_to_annotation_ids: HashMap<u32, Range<usize>>,
}

#[derive(Debug)]
pub struct COCO {
    coco_dataset: COCODetection,
    root_dir: PathBuf,
    coco_indices: COCOIndices,
}

impl COCO {
    pub fn new(root_dir: PathBuf, annotation_json_path: PathBuf) -> Result<Self, Error> {
        let file = File::open(annotation_json_path)?;

        let file_buffer = BufReader::new(file);

        let mut json_data: COCODetection = serde_json::from_reader(file_buffer)?;

        json_data.sort_images_inplace();
        json_data.sort_annots_inplace();

        let indices = COCOIndices::new(&json_data);

        Ok(COCO {
            coco_dataset: json_data,
            root_dir: root_dir,
            coco_indices: indices,
        })
    }
}

impl COCOIndices {
    fn new(coco_detections: &COCODetection) -> Self {
        let mut hashmap: HashMap<u32, Range<usize>> =
            HashMap::with_capacity(coco_detections.iter_images().len());

        let mut current_image_id: Option<u32> = None;
        let mut current_image_id_start: Option<usize> = None;
        let mut current_image_id_end: Option<usize> = None;

        let mut curr_index: usize = 0;
        for coco_annotation in coco_detections.iter_annotations() {
            if current_image_id.is_none() {
                current_image_id = Some(coco_annotation.image_id);
                current_image_id_start = Some(curr_index);
                current_image_id_end = Some(curr_index);
                curr_index += 1;
                continue;
            }

            if current_image_id.unwrap() == coco_annotation.image_id {
                current_image_id_end = Some(current_image_id_end.unwrap() + 1);
            } else {
                hashmap.insert(
                    current_image_id.unwrap(),
                    Range {
                        start: current_image_id_start.unwrap(),
                        end: current_image_id_end.unwrap() + 1,
                    },
                );

                current_image_id = Some(coco_annotation.image_id);
                current_image_id_start = Some(curr_index);
                current_image_id_end = Some(curr_index);
            }

            curr_index += 1;
        }

        hashmap.insert(
            current_image_id.unwrap(),
            Range {
                start: current_image_id_start.unwrap(),
                end: current_image_id_end.unwrap() + 1,
            },
        );

        COCOIndices {
            image_id_to_annotation_ids: hashmap,
        }
    }
}
