use std::{path::PathBuf, slice::Iter};

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
    pub filename: Option<PathBuf>,
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

#[derive(Debug, Deserialize)]
pub struct COCOPrediction {
    image_id: u32,
    category_id: u32,
    bbox: [f32; 4],
    score: f32,
    area: f32,
}

impl COCODetection {
    pub fn iter_images(&self) -> Iter<'_, COCOImage> {
        self.images.iter()
    }

    pub fn get_image_at_index(&self, idx: usize) -> Option<&COCOImage> {
        self.images.get(idx)
    }
    pub fn get_annotation_at_index(&self, idx: usize) -> Option<&Annotation> {
        self.annotations.get(idx)
    }
    pub fn get_category_at_index(&self, idx: usize) -> Option<&COCOCategory> {
        self.categories.get(idx)
    }

    pub fn sort_images_inplace(&mut self) {
        self.images.sort_unstable_by_key(|coco_image| coco_image.id);
    }

    pub fn sort_annots_inplace(&mut self) {
        self.annotations
            .sort_unstable_by_key(|coco_annotation| coco_annotation.id);
    }

    pub fn sort_categories_inplace(&mut self) {
        self.categories
            .sort_unstable_by_key(|coco_category| coco_category.id);
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

pub trait COCOEntry {
    type Id;

    fn get_id(&self) -> Self::Id;
}

impl COCOEntry for COCOImage {
    type Id = u32;
    fn get_id(&self) -> Self::Id {
        self.id
    }
}

impl COCOEntry for Annotation {
    type Id = u128;
    fn get_id(&self) -> Self::Id {
        self.id
    }
}

impl COCOEntry for COCOCategory {
    type Id = u32;
    fn get_id(&self) -> Self::Id {
        self.id
    }
}

impl<'a> COCOEntry for &'a COCOImage {
    type Id = u32;
    fn get_id(&self) -> Self::Id {
        self.id
    }
}

impl<'a> COCOEntry for &'a Annotation {
    type Id = u128;
    fn get_id(&self) -> Self::Id {
        self.id
    }
}

impl<'a> COCOEntry for &'a COCOCategory {
    type Id = u32;
    fn get_id(&self) -> Self::Id {
        self.id
    }
}
