"""
@author: qzz
@contact:q873264077@gmail.com
@version: 1.0.0
@file: compute_sl_metric.py
@time: 2024/1/12 14:42
"""
import os
import pprint

import numpy as np
import torch
import torchmetrics
from matplotlib import pyplot as plt

from set_path import append_sys_path

append_sys_path()
import bridge


def get_metrics(probs: torch.Tensor, labels: torch.Tensor, save_dir):
    pred_actions = torch.argmax(probs, 1)
    accuracy = (pred_actions == labels).int().sum() / pred_actions.shape[0]

    confusion_matrix = torchmetrics.functional.classification.multiclass_confusion_matrix(pred_actions,
                                                                                          labels, bridge.NUM_CALLS,
                                                                                          "none")

    # plot_confusion_matrix(confusion_matrix.numpy(), classes_str, True)

    stats = torchmetrics.functional.classification.multiclass_stat_scores(pred_actions, labels, bridge.NUM_CALLS,
                                                                          "none")

    # Compute TP, FP, TN, and FN for each class
    tp = stats[:, 0].squeeze()
    fp = stats[:, 1].squeeze()
    tn = stats[:, 2].squeeze()
    fn = stats[:, 3].squeeze()
    label_counts = stats[:, 4].squeeze()

    # p, r, f1 for each class

    precision = torchmetrics.functional.classification.multiclass_precision(pred_actions, labels, bridge.NUM_CALLS,
                                                                            "none")
    recall = torchmetrics.functional.classification.multiclass_recall(pred_actions, labels, bridge.NUM_CALLS, "none")
    f1_score = torchmetrics.functional.classification.multiclass_f1_score(pred_actions, labels, bridge.NUM_CALLS,
                                                                          "none")
    tick_marks = np.arange(0, 38)
    classes_str = ["Pass", "Dbl", "RDbl", "1C", "1D", "1H", "1S", "1NT", "2C", "2D", "2H", "2S", "2NT",
                   "3C", "3D", "3H", "3S", "3NT", "4C", "4D", "4H", "4S", "4NT",
                   "5C", "5D", "5H", "5S", "5NT", "6C", "6D", "6H", "6S", "6NT",
                   "7C", "7D", "7H", "7S", "7NT"]

    plt.figure(figsize=(5, 4))
    plt.plot(np.arange(0, bridge.NUM_CALLS), precision, color="blue", label="precision", linestyle="--")
    plt.xlabel("Ordered bids")
    # plt.title("Precision")
    plt.ylim(0, 1)
    # plt.xticks(tick_marks, classes_str, rotation=45)
    # plt.savefig(os.path.join(save_dir, "precision.png"))
    # plt.close()
    # plt.figure()
    plt.plot(np.arange(0, bridge.NUM_CALLS), recall, color="red", label="recall")
    plt.xlabel("Ordered calls")
    plt.ylim(0, 1)
    # plt.title("Recall")
    # plt.xticks(tick_marks, classes_str, rotation=45)
    # plt.show()
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "precision_and_recall.svg"), format="svg")
    plt.close()
    plt.figure()
    plt.plot(np.arange(0, bridge.NUM_CALLS), f1_score)
    plt.xlabel("Ordered Calls")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1)
    # plt.title("F1-Score")
    # plt.xticks(tick_marks, classes_str, rotation=45)
    plt.show()
    # plt.savefig(os.path.join(save_dir, "f1-score.png"))
    plt.close()

    # marco p, r, f1
    macro_averaged_precision = torchmetrics.functional.classification.multiclass_precision(pred_actions, labels,
                                                                                           bridge.NUM_CALLS, "macro")
    macro_averaged_recall = torchmetrics.functional.classification.multiclass_recall(pred_actions, labels,
                                                                                     bridge.NUM_CALLS, "macro")
    macro_averaged_f1 = torchmetrics.functional.classification.multiclass_f1_score(pred_actions, labels,
                                                                                   bridge.NUM_CALLS, "macro")

    # micro p, r, f1
    micro_averaged_precision = torchmetrics.functional.classification.multiclass_precision(pred_actions, labels,
                                                                                           bridge.NUM_CALLS, "micro")

    micro_averaged_recall = torchmetrics.functional.classification.multiclass_recall(pred_actions, labels,
                                                                                     bridge.NUM_CALLS, "micro")
    micro_averaged_f1 = torchmetrics.functional.classification.multiclass_f1_score(pred_actions, labels,
                                                                                   bridge.NUM_CALLS, "micro")

    # auc for each class
    auc_per_class = torchmetrics.functional.classification.multiclass_auroc(probs, labels, bridge.NUM_CALLS, "none")
    plt.figure()
    plt.plot(np.arange(0, 38), auc_per_class)
    plt.xlabel("Ordered Calls")
    plt.ylabel("AUC")
    # plt.title("AUC for all calls")
    plt.ylim(0.99, 1)
    # # plt.xticks(tick_marks, classes_str, rotation=45)
    plt.show()
    # plt.savefig(os.path.join(save_dir, "auc.png"))
    # plt.close()
    accuracy2 = torchmetrics.functional.classification.multiclass_accuracy(probs, labels, bridge.NUM_CALLS, "none")
    accuracy3 = torchmetrics.functional.classification.multiclass_accuracy(probs, labels, bridge.NUM_CALLS,
                                                                           average="micro")

    stats = {
        "accuracy": accuracy,
        "accuracy2": accuracy2,
        "accuracy3": accuracy3,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "label_counts": label_counts,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "micro_precision": micro_averaged_precision,
        "micro_recall": micro_averaged_recall,
        "micro_f1": micro_averaged_f1,
        "macro_precision": macro_averaged_precision,
        "macro_recall": macro_averaged_recall,
        "macro_f1": macro_averaged_f1,
        "auc": auc_per_class,
        "confusion_matrix": confusion_matrix
    }
    # torch.save(stats, os.path.join(save_dir, "stats.pth"))
    pprint.pprint(stats)
    return stats
