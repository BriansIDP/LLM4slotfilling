import argparse
import logging

from progress.bar import Bar

from metrics import ErrorMetric
from util import format_results, load_predictions, load_gold_data

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

logger = logging.getLogger(__name__)


def print_results(results):
    print(format_results(results=results,
                         label="intent (scen_act)",
                         full=args.full,
                         errors=args.errors,
                         table_layout=args.table_layout), "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SLURP evaluation script')
    parser.add_argument(
        '-g',
        '--gold-data',
        required=True,
        type=str,
        help='Gold data in SLURP jsonl format'
    )
    parser.add_argument(
        '-p1',
        '--prediction-file-sys1',
        type=str,
        required=True,
        help='Predictions file'
    )
    parser.add_argument(
        '-p2',
        '--prediction-file-sys2',
        type=str,
        required=True,
        help='Predictions file'
    )
    parser.add_argument(
        '--load-gold',
        action="store_true",
        help='When evaluating against gold transcriptions (gold_*_predictions.jsonl), this flag must be true.'
    )
    parser.add_argument(
        '--average',
        type=str,
        default='micro',
        help='The averaging modality {micro, macro}.'
    )
    parser.add_argument(
        '--full',
        action="store_true",
        help='Print the full results, including per-label metrics.'
    )
    parser.add_argument(
        '--errors',
        action="store_true",
        help='Print TPs, FPs, and FNs in each row.'
    )
    parser.add_argument(
        '--table-layout',
        type=str,
        default='fancy_grid',
        help='The results table layout {fancy_grid (DEFAULT), csv, tsv}.'
    )

    args = parser.parse_args()

    logger.info("Loading data")
    pred_examples_s1 = load_predictions(args.prediction_file_sys1, args.load_gold)
    pred_examples_s2 = load_predictions(args.prediction_file_sys2, args.load_gold)
    gold_examples = load_gold_data(args.gold_data, args.load_gold)
    n_gold_examples = len(gold_examples)

    logger.info("Initializing metrics")
    intent_f1_sys1 = ErrorMetric.get_instance(metric="f1", average=args.average)
    intent_f1_sys2 = ErrorMetric.get_instance(metric="f1", average=args.average)
    distance_metrics_sys1 = {}
    distance_metrics_sys2 = {}
    for distance in ['word', 'char']:
        distance_metrics_sys1[distance] = ErrorMetric.get_instance(metric="span_distance_f1",
                                                                   average=args.average,
                                                                   distance=distance)
        distance_metrics_sys2[distance] = ErrorMetric.get_instance(metric="span_distance_f1",
                                                                   average=args.average,
                                                                   distance=distance)
    slu_f1_sys1 = ErrorMetric.get_instance(metric="slu_f1", average=args.average)
    slu_f1_sys2 = ErrorMetric.get_instance(metric="slu_f1", average=args.average)

    for gold_id in list(gold_examples):
        if gold_id in pred_examples_s1 and gold_id in pred_examples_s2:
            gold_example = gold_examples.pop(gold_id)
            pred_example_s1 = pred_examples_s1.pop(gold_id)
            pred_example_s2 = pred_examples_s2.pop(gold_id)
            intent_f1_sys1("{}_{}".format(gold_example["scenario"], gold_example["action"]),
                           "{}_{}".format(pred_example_s1["scenario"], pred_example_s1["action"]))
            results1 = intent_f1_sys1.get_metric()
            intent_f1_sys2("{}_{}".format(gold_example["scenario"], gold_example["action"]),
                           "{}_{}".format(pred_example_s2["scenario"], pred_example_s2["action"]))
            results2 = intent_f1_sys2.get_metric()
            
            # if results1['overall'][5] > results2['overall'][5]:
            #     print(gold_id)
            # span_f1(gold_example["entities"], pred_example["entities"])
            for distance, metric in distance_metrics_sys1.items():
                metric(gold_example["entities"], pred_example_s1["entities"])
                slot_results1 = metric.get_metric()
                slu_f1_sys1(slot_results1)
            for distance, metric in distance_metrics_sys2.items():
                metric(gold_example["entities"], pred_example_s2["entities"])
                slot_results2 = metric.get_metric()
                slu_f1_sys2(slot_results2)
            slot_results1 = slu_f1_sys1.get_metric()
            slot_results2 = slu_f1_sys2.get_metric()
            if slot_results1['overall'][5] < slot_results2['overall'][5]:
                print(gold_id)
            for distance, metric in distance_metrics_sys1.items():
                metric.reset()
            for distance, metric in distance_metrics_sys2.items():
                metric.reset()
            slu_f1_sys1.reset()
            slu_f1_sys2.reset()
            intent_f1_sys1.reset()
            intent_f1_sys2.reset()
        else:
            raise Exception("{} not found".format(gold_id))

    # logger.info("Results:")
    # results = scenario_f1.get_metric()
    # print(format_results(results=results,
    #                      label="scenario",
    #                      full=args.full,
    #                      errors=args.errors,
    #                      table_layout=args.table_layout), "\n")

    # results = action_f1.get_metric()
    # print(format_results(results=results,
    #                      label="action",
    #                      full=args.full,
    #                      errors=args.errors,
    #                      table_layout=args.table_layout), "\n")

    # results = intent_f1.get_metric()
    # print(format_results(results=results,
    #                      label="intent (scen_act)",
    #                      full=args.full,
    #                      errors=args.errors,
    #                      table_layout=args.table_layout), "\n")

    # results = span_f1.get_metric()
    # print(format_results(results=results,
    #                      label="entities",
    #                      full=args.full,
    #                      errors=args.errors,
    #                      table_layout=args.table_layout), "\n")

    # for distance, metric in distance_metrics.items():
    #     results = metric.get_metric()
    #     slu_f1(results)
    #     print(format_results(results=results,
    #                          label="entities (distance {})".format(distance),
    #                          full=args.full,
    #                          errors=args.errors,
    #                          table_layout=args.table_layout), "\n")
    # results = slu_f1.get_metric()
    # print(format_results(results=results,
    #                      label="SLU F1",
    #                      full=args.full,
    #                      errors=args.errors,
    #                      table_layout=args.table_layout), "\n")

    # logger.warning("Gold examples not predicted: {} (out of {})".format(len(gold_examples), n_gold_examples))
