mod coco;
use std::{path::PathBuf, str::FromStr};

use coco::COCO;
use log::{error, info};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let ann_file =
        "/home/dg084/datasets/obj-det-dataset/coco/images/annotations/stuff_val2017.json";

    let root_dir = "/home/dg084/datasets/obj-det-dataset/coco/images/val2017";

    let annotation_path = PathBuf::from(ann_file);

    let root_dir_path = PathBuf::from(&root_dir);

    let coco = COCO::new(root_dir_path, annotation_path)?;

    info!("Done");

    Ok(())
}
