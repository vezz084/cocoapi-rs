mod coco;
use std::{path::PathBuf, str::FromStr};

use anyhow::Result;
use coco::COCO;
use log::{error, info};

fn main() -> Result<()> {
    env_logger::init();

    let ann_file = "./test_assets/stuff_val2017.json";

    let root_dir = "./test_assets/val_2017";

    let annotation_path = PathBuf::from(ann_file);

    let root_dir_path = PathBuf::from(&root_dir);

    let coco = COCO::new(root_dir_path, annotation_path)?;

    info!("Done");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    static COCO_INSTANCE: OnceLock<COCO> = OnceLock::new();

    fn get_shared_coco() -> &'static COCO {
        COCO_INSTANCE.get_or_init(|| {
            let ann_file = "./test_assets/stuff_val2017.json";
            let root_dir = ".";

            COCO::new(PathBuf::from(root_dir), PathBuf::from(ann_file))
                .expect("Failed to load global test COCO dataset")
        })
    }

    // --- CONSTRUCTOR TESTS ---

    #[test]
    fn test_coco_initialization_fails_gracefully() {
        let bad_path = PathBuf::from("./non_existent_file.json");
        let result = COCO::new(PathBuf::from("."), bad_path);
        assert!(
            result.is_err(),
            "Initialization should fail for non-existent files"
        );
    }

    // --- GET IMAGES TESTS ---

    #[test]
    fn test_get_images_happy_path() {
        let coco = get_shared_coco();
        let result = coco.get_images_from_ids(&[139]);

        assert_eq!(result.len(), 1);
        assert!(result[0].is_some(), "Expected to find image with ID 139");
    }

    #[test]
    fn test_get_images_edge_cases() {
        let coco = get_shared_coco();

        // Non-existent ID returns None
        let result = coco.get_images_from_ids(&[0]);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_none(), "ID 0 should return None");

        // Empty query returns empty vector
        assert!(coco.get_images_from_ids(&[]).is_empty());
    }

    // --- GET ANNOTATIONS TESTS ---

    #[test]
    fn test_get_annotations_happy_path() {
        let coco = get_shared_coco();
        let result = coco.get_annotations_from_ids(&[20000000]);

        assert_eq!(result.len(), 1);
        assert!(
            result[0].is_some(),
            "Expected to find annotation with ID 20000000"
        );
    }

    #[test]
    fn test_get_annotations_edge_cases() {
        let coco = get_shared_coco();

        let result = coco.get_annotations_from_ids(&[0]);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_none(), "ID 0 should return None");

        assert!(coco.get_annotations_from_ids(&[]).is_empty());
    }

    // --- GET CATEGORIES TESTS ---

    #[test]
    fn test_get_categories_happy_path() {
        let coco = get_shared_coco();
        let result = coco.get_categories_from_ids(&[180, 92]);

        assert_eq!(result.len(), 2);
        assert!(result[0].is_some(), "Expected to find category with ID 180");
        assert!(result[1].is_some(), "Expected to find category with ID 92");
    }

    #[test]
    fn test_get_categories_edge_cases() {
        let coco = get_shared_coco();

        let result = coco.get_categories_from_ids(&[0]);
        assert_eq!(result.len(), 1);
        assert!(result[0].is_none(), "ID 0 should return None");

        assert!(coco.get_categories_from_ids(&[]).is_empty());
    }

    // --- POSSIBLE ANNOTATIONS FROM IMAGES ---

    #[test]
    fn test_get_possible_annotations_from_image_ids() {
        let coco = get_shared_coco();

        assert!(coco.get_possible_annotations_from_image_ids(&[]).is_none());
        assert!(coco.get_possible_annotations_from_image_ids(&[0]).is_none());
        assert!(
            coco.get_possible_annotations_from_image_ids(&[139])
                .is_some()
        );
        assert!(
            coco.get_possible_annotations_from_image_ids(&[139, 0])
                .is_some()
        );
    }

    // --- POSSIBLE ANNOTATIONS FROM CATEGORIES ---

    #[test]
    fn test_get_possible_annotations_from_category_ids() {
        let coco = get_shared_coco();

        assert!(
            coco.get_possible_annotations_from_category_ids(&[])
                .is_none()
        );
        assert!(
            coco.get_possible_annotations_from_category_ids(&[0])
                .is_none()
        );

        let res_single = coco.get_possible_annotations_from_category_ids(&[92]);
        assert_eq!(
            res_single.as_ref().map(|v| v.len()),
            Some(141),
            "Category 92 should yield exactly 141 annotations"
        );

        let res_multi = coco.get_possible_annotations_from_category_ids(&[92, 0, 93]);
        assert_eq!(
            res_multi.as_ref().map(|v| v.len()),
            Some(269),
            "Categories [92, 0, 93] should yield exactly 269 annotations"
        );
    }

    // --- ANNOTATIONS FROM FILTERS ---

    #[test]
    fn test_get_possible_annotations_from_filters_structural_nones() {
        let coco = get_shared_coco();

        assert!(
            coco.get_possible_annotations_from_filters(None, None)
                .is_none()
        );
        assert!(
            coco.get_possible_annotations_from_filters(None, Some(&[]))
                .is_none()
        );
        assert!(
            coco.get_possible_annotations_from_filters(Some(&[]), None)
                .is_none()
        );
        assert!(
            coco.get_possible_annotations_from_filters(Some(&[]), Some(&[]))
                .is_none()
        );
    }

    #[test]
    fn test_get_possible_annotations_from_filters_categories() {
        let coco = get_shared_coco();

        assert!(
            coco.get_possible_annotations_from_filters(None, Some(&[0]))
                .is_none()
        );
        assert!(
            coco.get_possible_annotations_from_filters(Some(&[0]), Some(&[0]))
                .is_none()
        );

        let filter_cat_len_1 = coco.get_possible_annotations_from_filters(None, Some(&[0, 92]));
        assert_eq!(filter_cat_len_1.as_ref().map(|v| v.len()), Some(141));

        let filter_cat_len_2 = coco.get_possible_annotations_from_filters(None, Some(&[0, 92, 93]));
        assert_eq!(filter_cat_len_2.as_ref().map(|v| v.len()), Some(269));
    }

    #[test]
    fn test_get_possible_annotations_from_filters_images() {
        let coco = get_shared_coco();

        assert!(
            coco.get_possible_annotations_from_filters(Some(&[0]), None)
                .is_none()
        );

        let filter_img_len_1 = coco.get_possible_annotations_from_filters(Some(&[0, 8211]), None);
        assert_eq!(filter_img_len_1.as_ref().map(|v| v.len()), Some(11));

        let filter_img_len_2 =
            coco.get_possible_annotations_from_filters(Some(&[0, 92, 8211]), None);
        assert_eq!(filter_img_len_2.as_ref().map(|v| v.len()), Some(11));
    }

    #[test]
    fn test_get_possible_annotations_from_filters_mixed() {
        let coco = get_shared_coco();

        let mixed_len =
            coco.get_possible_annotations_from_filters(Some(&[0, 92, 8211]), Some(&[92, 96]));
        assert_eq!(mixed_len.as_ref().map(|v| v.len()), Some(2));

        let mixed_len_with_zero =
            coco.get_possible_annotations_from_filters(Some(&[0, 92, 8211]), Some(&[92, 96, 0]));
        assert_eq!(mixed_len_with_zero.as_ref().map(|v| v.len()), Some(2));

        assert!(
            coco.get_possible_annotations_from_filters(Some(&[0, 92, 8211]), Some(&[0]))
                .is_none()
        );
    }

    // --- IMAGES FROM FILTERS ---

    #[test]
    fn test_get_possible_images_from_filters_structural_nones() {
        let coco = get_shared_coco();

        assert!(coco.get_possible_images_from_filters(None, None).is_none());
        assert!(
            coco.get_possible_images_from_filters(None, Some(&[]))
                .is_none()
        );
        assert!(
            coco.get_possible_images_from_filters(Some(&[]), None)
                .is_none()
        );
        assert!(
            coco.get_possible_images_from_filters(Some(&[]), Some(&[]))
                .is_none()
        );
    }

    #[test]
    fn test_get_possible_images_from_filters_categories() {
        let coco = get_shared_coco();

        assert!(
            coco.get_possible_images_from_filters(None, Some(&[0]))
                .is_none()
        );
        assert!(
            coco.get_possible_images_from_filters(Some(&[0]), Some(&[0]))
                .is_none()
        );

        let filter_cat_len_1 = coco.get_possible_images_from_filters(None, Some(&[0, 92]));
        assert_eq!(filter_cat_len_1.as_ref().map(|v| v.len()), Some(141));

        let filter_cat_len_2 = coco.get_possible_images_from_filters(None, Some(&[0, 92, 93]));
        assert_eq!(filter_cat_len_2.as_ref().map(|v| v.len()), Some(269));
    }

    #[test]
    fn test_get_possible_images_from_filters_images() {
        let coco = get_shared_coco();

        assert!(
            coco.get_possible_images_from_filters(Some(&[0]), None)
                .is_none()
        );

        let filter_img_len_1 = coco.get_possible_images_from_filters(Some(&[0, 8211]), None);
        assert_eq!(filter_img_len_1.as_ref().map(|v| v.len()), Some(1));

        let filter_img_len_2 = coco.get_possible_images_from_filters(Some(&[0, 92, 8211]), None);
        assert_eq!(filter_img_len_2.as_ref().map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_get_possible_images_from_filters_mixed() {
        let coco = get_shared_coco();

        let mixed_len =
            coco.get_possible_images_from_filters(Some(&[0, 92, 8211]), Some(&[92, 96]));
        assert_eq!(mixed_len.as_ref().map(|v| v.len()), Some(1));

        let mixed_len_with_zero =
            coco.get_possible_images_from_filters(Some(&[0, 92, 8211]), Some(&[92, 96, 0]));
        assert_eq!(mixed_len_with_zero.as_ref().map(|v| v.len()), Some(1));

        assert!(
            coco.get_possible_images_from_filters(Some(&[0, 92, 8211]), Some(&[0]))
                .is_none()
        );
    }
}
