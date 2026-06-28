use std::slice::Iter;

use serde::Deserialize;
#[derive(Deserialize, Debug)]
pub struct COCOCategory {
    pub supercategory: String,
    pub id: u32,
    pub name: String,
}
#[derive(Deserialize, Debug)]
pub struct Segmentation {
    pub counts: String,
    pub size: [u32; 2],
}
#[derive(Deserialize, Debug)]
pub struct Annotation {
    pub area: f32,
    pub iscrowd: u8,
    pub image_id: u32,
    pub bbox: [f32; 4],
    pub category_id: u32,
    pub id: u128,
}
#[derive(Deserialize, Debug)]
pub struct COCOImage {
    pub license: Option<u32>,
    pub coco_url: Option<String>,
    pub height: u32,
    pub width: u32,
    pub data_captured: Option<String>,
    pub flickr_url: Option<String>,
    pub id: u32,
}

#[derive(Deserialize, Debug)]
pub struct COCODetection {
    images: Vec<COCOImage>,
    annotations: Vec<Annotation>,
    categories: Vec<COCOCategory>,
}

impl COCODetection {
    pub fn iter_images(&self) -> Iter<'_, COCOImage> {
        self.images.iter()
    }

    pub fn iter_annotations(&self) -> Iter<'_, Annotation> {
        self.annotations.iter()
    }

    pub fn iter_categories(&self) -> Iter<'_, COCOCategory> {
        self.categories.iter()
    }

    pub fn get_all_image_ids(&self) -> impl Iterator<Item = &u32> {
        self.iter_images().map(|coco_image| &coco_image.id)
    }

    pub fn get_all_annotation_ids(&self) -> impl Iterator<Item = &u128> {
        self.iter_annotations()
            .map(|coco_annotation| &coco_annotation.id)
    }

    pub fn get_annotations_from_image_ids(
        &self,
        image_ids: &[u32],
    ) -> impl Iterator<Item = &Annotation> {
        self.iter_annotations()
            .filter(|coco_annotation| image_ids.contains(&coco_annotation.image_id))
    }
}
