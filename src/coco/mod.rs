pub mod coco_types;

use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Error},
    path::{Path, PathBuf},
};

pub use coco_types::*;

#[derive(Debug)]
pub struct COCO {
    coco_dataset: COCODetection,
    root_dir: PathBuf,
}

struct COCOIndices<'a> {
    image_id_to_filename: HashMap<&'a u32, PathBuf>,
    image_id_to_annotation_ids: HashMap<&'a u32, Vec<&'a Annotation>>,
}

impl COCO {
    pub fn new(root_dir: PathBuf, annotation_json_path: PathBuf) -> Result<Self, Error> {
        let file = File::open(annotation_json_path)?;

        let file_buffer = BufReader::new(file);

        let json_data: COCODetection = serde_json::from_reader(file_buffer)?;

        COCOIndices::new(&json_data, &root_dir);
        Ok(COCO {
            coco_dataset: json_data,
            root_dir: root_dir,
        })
    }
}

impl<'a> COCOIndices<'a> {
    fn new(coco_detections: &'a COCODetection, root_dir: &Path) -> Self {
        let coco_images = coco_detections.iter_images();
        let mut id_to_img_path: HashMap<&u32, PathBuf> = HashMap::with_capacity(coco_images.len());

        for coco_image in coco_images {
            id_to_img_path.insert(
                &coco_image.id,
                root_dir.join(coco_image.filename.as_ref().unwrap()),
            );
        }

        let coco_images = coco_detections.iter_images();

        let mut image_id_to_annotations: HashMap<&u32, Vec<&Annotation>> =
            HashMap::with_capacity(coco_images.len());

        for coco_image in coco_images {
            let annots = coco_detections
                .get_annotations_from_image_ids(&[coco_image.id])
                .collect();
            image_id_to_annotations.insert(&coco_image.id, annots);
        }

        COCOIndices {
            image_id_to_filename: id_to_img_path,
            image_id_to_annotation_ids: image_id_to_annotations,
        }
    }
}
