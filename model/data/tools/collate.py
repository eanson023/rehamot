from typing import List, Dict

from torch import Tensor


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_datastruct_and_text(lst_elements: List) -> Dict:
    collate_datastruct = lst_elements[0]["motion"].transforms.collate

    batch = {
        # Collate with padding for the motion
        # only use features
        "motion": collate_datastruct([x["motion"] for x in lst_elements]).features,
        # Collate normally for the length
        "length": [x["length"] for x in lst_elements],
        # Collate the text
        "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch


def collate_datastruct_and_text_v2(lst_elements: List) -> Dict:
    # Sort a data list by caption length
    # lst_elements.sort(key=lambda x: len(x["text"]), reverse=True)

    batch = {
        # Collate with padding for the motion
        "motion": collate_tensor_with_padding([x["motion"] for x in lst_elements]),
        # Collate normally for the length
        "length": [x["length"] for x in lst_elements],
        # Collate the text
        "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch


def collate_text_and_length(lst_elements: Dict) -> Dict:
    batch = {"length": [x["length"] for x in lst_elements],
             "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys(
    ) if x not in batch and x != "motion"]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]
    return batch
