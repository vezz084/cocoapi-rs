pub mod coco_types;

use std::{
    fs::File,
    io::{BufReader, Error},
    path::PathBuf,
};

pub use coco_types::*;

pub struct COCO {
    coco_dataset: COCODetection,
    root_dir: PathBuf,
}

impl COCO {
    pub fn new(root_dir: PathBuf, annotation_json_path: PathBuf) -> Result<Self, Error> {
        let file = File::open(annotation_json_path)?;

        let file_buffer = BufReader::new(file);

        let json_data: COCODetection = serde_json::from_reader(file_buffer)?;

        Ok(COCO {
            coco_dataset: json_data,
            root_dir: root_dir,
        })
    }
}
