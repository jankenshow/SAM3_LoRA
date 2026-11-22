"""Convert model-independent Samples into SAM3 types at the runtime boundary."""

from sam3_lora.modeling import ensure_sam3_path


def to_datapoint(sample):
    ensure_sam3_path()
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image,
        InferenceMetadata,
        Object,
    )

    height, width = sample.original_size
    image = Image(
        data=sample.image,
        objects=[
            Object(
                bbox=o.bbox,
                area=float(o.bbox[2] * o.bbox[3]),
                object_id=o.object_id,
                segment=o.segment,
            )
            for o in sample.objects
        ],
        size=tuple(sample.image.shape[-2:]),
    )
    queries = [
        FindQueryLoaded(
            query_text=q.text,
            image_id=0,
            object_ids_output=list(q.object_ids),
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=sample.source_id,
                original_image_id=sample.source_index,
                original_category_id=-1 if q.negative else 0,
                original_size=(height, width),
                object_id=-1,
                frame_index=-1,
            ),
        )
        for q in sample.queries
    ]
    return Datapoint(find_queries=queries, images=[image])


def collate_samples(samples):
    ensure_sam3_path()
    from sam3.train.data.collator import collate_fn_api

    return collate_fn_api([to_datapoint(s) for s in samples], dict_key="input", with_seg_masks=True)


def final_output(outputs):
    ensure_sam3_path()
    from sam3.model.model_misc import SAM3Output

    with SAM3Output.iteration_mode(
        outputs, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
    ) as stages:
        return list(stages)[-1][-1]
