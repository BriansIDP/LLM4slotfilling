import argparse
import logging
import json

from progress.bar import Bar

from metrics import ErrorMetric
from util import format_results, load_predictions, load_gold_data

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

logger = logging.getLogger(__name__)

def is_unseen(worddict, entity):
    for word in entity.split():
        if word not in worddict:
            return True
    return False

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
        '-p',
        '--prediction-file',
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
    parser.add_argument(
        '--heldout',
        type=str,
        default='',
        help='The list of heldout slots for evaluation'
    )
    parser.add_argument(
        '--freqfile',
        type=str,
        default='',
        help='The entity frequency bins',
    )

    args = parser.parse_args()

    unbounded = ['date', 'timeofday', 'time', 'general_frequency', 'change_amount', 'query_detail', 'weather_descriptor', 'music_descriptor']

    logger.info("Loading data")
    pred_examples = load_predictions(args.prediction_file, args.load_gold)
    gold_examples = load_gold_data(args.gold_data, args.load_gold)
    n_gold_examples = len(gold_examples)

    freqbins = {}
    if args.freqfile != '':
        freqbins = {'100-':[], '0-100':[]}
        with open(args.freqfile) as fin:
            freqs = json.load(fin)

    logger.info("Initializing metrics")
    scenario_f1 = ErrorMetric.get_instance(metric="f1", average=args.average)
    action_f1 = ErrorMetric.get_instance(metric="f1", average=args.average)
    intent_f1 = ErrorMetric.get_instance(metric="f1", average=args.average)
    span_f1 = ErrorMetric.get_instance(metric="span_f1", average=args.average)
    distance_metrics = {}
    for distance in ['word', 'char']:
        distance_metrics[distance] = ErrorMetric.get_instance(metric="span_distance_f1",
                                                              average=args.average,
                                                              distance=distance)
    slu_f1 = ErrorMetric.get_instance(metric="slu_f1", average=args.average)
    unseencount = 0
    corrected = 0
    totalcorr = 0

    # with open("/home/gs534/rds/hpc-work/work/slurp/data/KBs/wordfreq.txt") as fin:
    #     worddict = {line.split()[0]:int(line.split()[1]) for line in fin}
    # with open("/home/gs534/rds/hpc-work/work/slurp/data/KBs/word_onts/entlist_wordf30.txt") as fin:
    #     KBentlist = [line.strip().lower() for line in fin]

    extra = {}
    # bar = Bar(message="Evaluating metrics", max=len(gold_examples))
    for gold_id in list(gold_examples):
        if gold_id in pred_examples:
            gold_example = gold_examples.pop(gold_id)
            pred_example = pred_examples.pop(gold_id)
            scenario_f1(gold_example["scenario"], pred_example["scenario"])
            action_f1(gold_example["action"], pred_example["action"])
            intent_f1("{}_{}".format(gold_example["scenario"], gold_example["action"]),
                      "{}_{}".format(pred_example["scenario"], pred_example["action"]))
            for ent in gold_example["entities"]:
                ent['filler'] = ent['filler'].replace(" 's", "'s")
            if args.heldout != '':
                heldout = args.heldout.split()
                newgold = []
                newpred = []
                for ent in gold_example["entities"]:
                    if ent["type"] in heldout:
                        newgold.append(ent)
                for ent in pred_example["entities"]:
                    if ent["type"] in heldout:
                        newpred.append(ent)
                span_f1(newgold, newpred)
                for distance, metric in distance_metrics.items():
                    metric(newgold, newpred)
            elif freqbins != {}:
                newgold = []
                newpred = []
                thred_low = 0
                thred_high = 5
                for ent in gold_example["entities"]:
                    # if ent['filler'] in freqs and freqs[ent['filler']] > thred_low and freqs[ent['filler']] < thred_high:
                    if ent['filler'] not in freqs and ent['type'] not in unbounded and (is_unseen(worddict, ent['filler']) or len(ent['filler'].split()) > 1):
                    # if ent['filler'] not in freqs and ent['filler'] in KBentlist:
                        newgold.append(ent)
                        unseencount += 1
                    # if not is_unseen(worddict, ent['filler']) and ent['filler'] not in freqs and ent['type'] not in unbounded:
                    #     if ent['filler'] not in extra:
                    #         extra[ent['filler']] = ent['type']
                for ent in pred_example["entities"]:
                    # if (ent['filler'] in freqs and freqs[ent['filler']] > thred_low and freqs[ent['filler']] < thred_high):
                    if ent['filler'] not in freqs and ent['type'] not in unbounded and (is_unseen(worddict, ent['filler']) or len(ent['filler'].split()) > 1):
                    # if ent['filler'] not in freqs # and is_unseen(worddict, ent['filler']): # and ent['filler'] in KBentlist:
                        newpred.append(ent)
                # if newgold != [] and newpred != []:
                #     for entity in newgold:
                #         if entity in newpred:
                #             totalcorr += 1
                #             if entity["filler"] not in pred_example["text"]:
                #                 corrected += 1
                span_f1(newgold, newpred)
                for distance, metric in distance_metrics.items():
                    metric(newgold, newpred)
            else:
                span_f1(gold_example["entities"], pred_example["entities"])
                for distance, metric in distance_metrics.items():
                    metric(gold_example["entities"], pred_example["entities"])
        # bar.next()
    # bar.finish()

    logger.info("Results:")
    results = scenario_f1.get_metric()
    print(format_results(results=results,
                         label="scenario",
                         full=args.full,
                         errors=args.errors,
                         table_layout=args.table_layout), "\n")

    results = action_f1.get_metric()
    print(format_results(results=results,
                         label="action",
                         full=args.full,
                         errors=args.errors,
                         table_layout=args.table_layout), "\n")

    results = intent_f1.get_metric()
    print(format_results(results=results,
                         label="intent (scen_act)",
                         full=args.full,
                         errors=args.errors,
                         table_layout=args.table_layout), "\n")

    results = span_f1.get_metric()
    print(format_results(results=results,
                         label="entities",
                         full=args.full,
                         errors=args.errors,
                         table_layout=args.table_layout), "\n")

    for distance, metric in distance_metrics.items():
        results = metric.get_metric()
        slu_f1(results)
        print(format_results(results=results,
                             label="entities (distance {})".format(distance),
                             full=args.full,
                             errors=args.errors,
                             table_layout=args.table_layout), "\n")
    results = slu_f1.get_metric()
    print(format_results(results=results,
                         label="SLU F1",
                         full=args.full,
                         errors=args.errors,
                         table_layout=args.table_layout), "\n")
    print(unseencount)
    print("Corrected: {}/{}".format(corrected, totalcorr))

    logger.warning("Gold examples not predicted: {} (out of {})".format(len(gold_examples), n_gold_examples))

    # with open("extra_unseen.json", "w") as fout:
    #     json.dump(extra, fout, indent=4)
