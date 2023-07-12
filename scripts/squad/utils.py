#remove token_type_ids and p_mask
import re
import timeit
from tqdm import tqdm
import os, json
import logging
logger = logging.getLogger(__name__)
import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import SequentialSampler, DataLoader
from typing import List
from transformers.data.metrics.squad_metrics import (
  compute_predictions_log_probs,
  compute_predictions_logits,
  squad_evaluate,
)

from transformers.data.processors.squad import (
  SquadResult,
  SquadV1Processor,
  SquadV2Processor,
  squad_convert_examples_to_features
)

class MultilingualSQuADDataset(Dataset):
    def __init__(self, args, langs: List[str], split: str, prefix: str, tokenizer=None):
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        self.split = split

        max_seq_length  = args.max_seq_length

        self.lang_datasets = {}
        self.lang_features = {}
        self.lang_examples = {}

        self.test_files = None
        if split=='train':
            self.data_files = {'en': args.train_file}
        else:
            self.data_files = {'en':args.test_file}
        self.data_dir = os.path.dirname(args.train_file)

        self.cached_features_files = {lang : os.path.join(self.data_dir, f'{prefix}_{split}_{max_seq_length}_{lang}') for lang in langs}

        for lang, cached_features_file in self.cached_features_files.items():
            if os.path.exists(cached_features_file):
                logger.info("Loading features from cached file %s", cached_features_file)
                features_and_dataset = torch.load(cached_features_file)
                features, dataset, examples = features_and_dataset["features"], features_and_dataset["dataset"], features_and_dataset["examples"]
            else:
                logger.info("Creating features from dataset file at %s", cached_features_file)
                processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
                if split == 'train':
                    print (self.data_files[lang])
                    examples = processor.get_train_examples(self.data_dir, filename=self.data_files[lang])
                elif split == 'dev' or split=='test':
                    print (self.data_files[lang])
                    examples = processor.get_dev_examples(self.data_dir, filename=self.data_files[lang])
                else:
                    raise ValueError

                features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=(split=='train'),
                return_dataset="pt",
                threads=args.threads
                )

                if args.local_rank in [-1, 0]:
                    logger.info("Saving features into cached file %s", cached_features_file)
                    if split == 'train':
                        examples = None
                        features = None
                    torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)


            self.lang_datasets[lang] = dataset
            self.lang_features[lang] = features
            self.lang_examples[lang] = examples

        if args.local_rank == 0:
            torch.distributed.barrier()

        self.all_dataset = ConcatDataset(list(self.lang_datasets.values()))
            
    def __getitem__(self, index):
        return self.all_dataset[index]
    
    def __len__(self):
        return len(self.all_dataset)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def predict( model, eval_dataset, args, tokenizer, step, is_save_logits=False):
    lang_results = {}
    for lang in eval_dataset.lang_datasets:
        dataset = eval_dataset.lang_datasets[lang]
        examples = eval_dataset.lang_examples[lang]
        features = eval_dataset.lang_features[lang]
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} {}*****".format(step, lang))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_results = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2], #None if model_type in ["xlm", "distilbert", "xlmr",] else batch[2],
                }
                example_indices = batch[3]

                outputs = model(**inputs)[:2] #start logits and end logits

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)
        if is_save_logits:
            logger.info("Save logits")
            output_logits_file = os.path.join(args.output_dir, str(step), f"all_results-{lang}.logits")
            os.makedirs(os.path.dirname(output_logits_file),exist_ok=True)
            torch.save(all_results,output_logits_file)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

        # Compute predictions
        output_prediction_file = os.path.join(args.output_dir, str(step), f"test-{lang}.json")
        output_nbest_file = os.path.join(args.output_dir, str(step), f"nbest_predictions-{lang}.json")
        os.makedirs(os.path.dirname(output_prediction_file),exist_ok=True)
        if args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(args.output_dir, str(step), "null_odds_{}.json".format(step))
        else:
            output_null_log_odds_file = None

        predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False, #args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer
        )

        # Compute the F1 and exact scores.
        results = squad_evaluate(examples, predictions)
        logger.info("{} :Results: {}".format(lang, results))
        lang_results[lang] = results

    eval_results_file = os.path.join(args.output_dir, 'eval_results.txt')
    with open(eval_results_file,'a') as f:
        line = f'Step {step} -- '+ ' '.join([f"{lang}:{results['f1']:.1f}/{results['exact']:.1f}" for lang, results in lang_results.items()])
        avg_f1 = sum(results['f1'] for results in lang_results.values()) / len(lang_results)
        avg_em = sum(results['exact'] for results in lang_results.values()) / len(lang_results)
        line += f' avg:{avg_f1:.1f}/{avg_em:.1f}\n'
        f.write(line)

    return lang_results
