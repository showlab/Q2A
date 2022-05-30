import json
import numpy as np

def evaluate_for_scores(all_scores, all_labels):
    recall_1, recall_3, mean_rank, mean_reciprocal_rank = [], [], [], []
    for scores, label in zip(all_scores, all_labels):
        sorted_indices = scores.argsort()[::-1]
        mask = sorted_indices == label
        recall_1.append(float(mask[0]))
        recall_3.append(float(mask[:3].sum()))
        mean_rank.append(float(mask.nonzero()[0] + 1))
        mean_reciprocal_rank.append(len(mask) / (mean_rank[-1]))
    recall_1 = sum(recall_1) / len(recall_1)
    recall_3 = sum(recall_3) / len(recall_3)
    mean_rank = sum(mean_rank) / len(mean_rank)
    mean_reciprocal_rank = sum(mean_reciprocal_rank) / len(mean_reciprocal_rank)
    return recall_1, recall_3, mean_rank, mean_reciprocal_rank

def evaluate(all_preds, all_annos):
    all_scores, all_labels = [], []
    for key in all_annos:
        annos = all_annos[key]
        preds = all_preds[key]
        for anno in annos:
            # find the corresponding question in preds
            is_matched = False
            for pred in preds:
                if pred['question'] == anno['question']:
                    # calculate
                    is_matched = True
                    for scores_per_step, label_per_step, answer_per_step in zip(pred['scores'], anno['correct'], anno['answers']):
                        assert len(scores_per_step) == len(answer_per_step), f"Please check your submission file. The length of scores list {len(scores_per_step)} should be equal to length of candidate answers {len(answer_per_step)}."
                        all_scores.append(np.array(scores_per_step))
                        all_labels.append(np.array(label_per_step - 1)) # label starts from 0
                    break
        assert is_matched, f"Please check your submission file. We cant find predictions for the question: {anno['question']} (data folder: {key})."

    recall_1, recall_3, mean_rank, mean_reciprocal_rank = evaluate_for_scores(all_scores, all_labels)
    return recall_1, recall_3, mean_rank, mean_reciprocal_rank


if __name__ == "__main__":
    # participates' results
    with open("submit_test.json") as f:
        all_preds = json.load(f)

    # ground-truth annotations (participants dont have now)
    with open("/data/chenjoya/assistq/test_with_gt.json") as f:
        all_annos = json.load(f)

    evaluate(all_preds, all_annos)
