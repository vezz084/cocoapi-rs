mod coco;
use std::{path::PathBuf, str::FromStr};

use coco::COCO;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let annotation_path = PathBuf::from_str(
        "/home/dg084/datasets/obj-det-dataset/coco/images/annotations/stuff_val2017.json",
    )?;

    let root_dir_path =
        PathBuf::from_str("/home/dg084/datasets/obj-det-dataset/coco/images/val2017")?;

    COCO::new(root_dir_path, annotation_path)?;

    Ok(())
}
