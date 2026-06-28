pub struct COCOCategory {
    pub supercategory: String,
    pub id: u16,
    pub name: String,
}

pub struct Segmentation {
    pub counts: String,
    pub size: [u16; 2],
}

pub struct Annotation {
    pub area: f32,
    pub iscrowd: bool,
    pub image_id: u16,
    pub bbox: [f32; 4],
    pub category_id: u16,
    pub id: u128,
}

pub struct COCOImage {
    pub license: Option<u16>,
    pub coco_url: Option<String>,
    pub height: u16,
    pub width: u16,
    pub data_captured: Option<String>,
    pub flickr_url: Option<String>,
    pub id: u16,
}
