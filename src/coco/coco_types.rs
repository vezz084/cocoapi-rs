use serde::Deserialize;
#[derive(Deserialize)]
pub struct COCOCategory {
    pub supercategory: String,
    pub id: u32,
    pub name: String,
}
#[derive(Deserialize)]
pub struct Segmentation {
    pub counts: String,
    pub size: [u32; 2],
}
#[derive(Deserialize)]
pub struct Annotation {
    pub area: f32,
    pub iscrowd: u8,
    pub image_id: u32,
    pub bbox: [f32; 4],
    pub category_id: u32,
    pub id: u128,
}
#[derive(Deserialize)]
pub struct COCOImage {
    pub license: Option<u32>,
    pub coco_url: Option<String>,
    pub height: u32,
    pub width: u32,
    pub data_captured: Option<String>,
    pub flickr_url: Option<String>,
    pub id: u32,
}

#[derive(Deserialize)]
pub struct COCODetection {
    images: Vec<COCOImage>,
    annotations: Vec<Annotation>,
    categories: Vec<COCOCategory>,
}
