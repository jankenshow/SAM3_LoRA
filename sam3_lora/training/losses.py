"""SAM3 matching/loss policy, separated from optimization and orchestration."""

from sam3_lora.modeling import ensure_sam3_path
from sam3_lora.utils.tensors import move_to_device


class SAM3LossStep:
    def __init__(self, device):
        ensure_sam3_path()
        from sam3.train.loss.loss_fns import Boxes, IABCEMdetr, Masks
        from sam3.train.loss.sam3_loss import Sam3LossWrapper
        from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

        self.device = device
        self.matcher = BinaryHungarianMatcherV2(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
        )
        self.wrapper = Sam3LossWrapper(
            loss_fns_find=[
                Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
                IABCEMdetr(
                    pos_weight=10.0,
                    weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                    pos_focal=False,
                    alpha=0.25,
                    gamma=2,
                    use_presence=True,
                    pad_n_queries=200,
                ),
                Masks(
                    weight_dict={"loss_mask": 200.0, "loss_dice": 10.0},
                    focal_alpha=0.25,
                    focal_gamma=2.0,
                    compute_aux=False,
                ),
            ],
            matcher=self.matcher,
            o2m_matcher=BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4),
            o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False,
            normalization="local",
            normalize_by_valid_object_num=False,
        )

    def __call__(self, model, batch):
        from sam3.model.model_misc import SAM3Output

        inputs = move_to_device(batch["input"], self.device)
        original = model.module if hasattr(model, "module") else model
        targets = move_to_device(
            [original.back_convert(t) for t in inputs.find_targets], self.device
        )
        outputs = model(inputs)
        with SAM3Output.iteration_mode(
            outputs, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as stages:
            for stage, target in zip(stages, targets):
                for step in stage:
                    step["indices"] = self.matcher(step, target)
                    for auxiliary in step.get("aux_outputs", []):
                        auxiliary["indices"] = self.matcher(auxiliary, target)
        return self.wrapper(outputs, targets)["core_loss"]
