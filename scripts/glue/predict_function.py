import numpy as np
import os
import torch
from torch.utils.data import SequentialSampler,DistributedSampler,DataLoader
from utils_glue import compute_metrics
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)


def predict(model,eval_datasets,step, eval_lang, args, taskname=None):
    eval_task = taskname
    eval_output_dir = args.output_dir
    lang_results = {}
    for lang,eval_dataset in zip(eval_lang, eval_datasets):
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        logger.info("Predicting...")
        logger.info("***** Running predictions *****")
        logger.info(" task name = %s", eval_task)
        logger.info(" lang : %s", lang)
        logger.info("  Num  examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.predict_batch_size)
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.predict_batch_size)
        model.eval()

        pred_logits = []
        label_ids = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=None):
            token_type_ids = batch.get('token_type_ids',None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(args.device)
            input_ids, input_mask, labels = batch['input_ids'],batch['attention_mask'],batch['label']
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            #segment_ids = segment_ids.to(args.device)
            with torch.no_grad():
                outputs= model(input_ids, input_mask,token_type_ids=token_type_ids)
                logits = outputs[0]
            pred_logits.append(logits.detach().cpu())
            label_ids.append(labels)
        pred_logits = np.array(torch.cat(pred_logits),dtype=np.float32)
        label_ids = np.array(torch.cat(label_ids),dtype=np.int64)

        preds = np.argmax(pred_logits, axis=1)
        results = compute_metrics(eval_task, preds, label_ids)

        logger.info("***** Eval results {} Lang {} *****".format(step, lang))
        for key in sorted(results.keys()):
            logger.info(f"{lang} {key} = {results[key]:.5f}")
        lang_results[lang] = results

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")

    write_results(output_eval_file,step,lang_results, eval_lang)
    model.train()
    return lang_results

def write_results(output_eval_file,step,lang_results, eval_lang):
    with open(output_eval_file, "a") as writer:
            writer.write(f"step: {step:<8d} ")
            line = "Acc/F1:"

            for lang in eval_lang:
                acc = lang_results[lang]['acc']
                if 'f1' in lang_results[lang]:
                    f1 = lang_results[lang]['f1']
                    line += f"{lang}={acc:.5f}/{f1:.5f} "
                else:
                    line += f"{lang}={acc:.5f} "
            writer.write(line+'\n')