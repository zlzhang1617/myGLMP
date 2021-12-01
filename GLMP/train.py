import argparse
import os
import sys
import time
import random
import logging
from tqdm import tqdm
from typing import Optional, List, Tuple

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import get_linear_schedule_with_warmup


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "GLMP/log/")
print(__file__.split("/")[-1] + ": " + BASE_DIR)
sys.path.append(BASE_DIR)


from common.data import Data
from GLMP.dataset import GLMP_Dataset, collate_fn
from GLMP.model import GLMP
from utils.lang import Lang
from utils.functions import compute_entity_PRF, get_global_entities, nltk_multi_bleu


t = time.strftime("%m-%d-%H:%M", time.localtime())
log_file = LOG_DIR + "main_" + t + ".log"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    filename=log_file,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="train.txt")
    parser.add_argument("--dev_file", default="dev.txt", type=str)
    parser.add_argument("--init_model_dir", default="", type=str)
    parser.add_argument("--model_name", type=str, default="GLMP")

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_steps", default=500, type=int)

    parser.add_argument("--memory_hop", default=3, type=int)
    parser.add_argument("--memory_len", default=6, type=int)
    parser.add_argument("--memory_size", default=500, type=int)
    parser.add_argument("--sent_len", default=300, type=int)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--save_checkpoints", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)

    args = parser.parse_args()
    return args


def train(
    model: GLMP,
    dataset: GLMP_Dataset,
    args: argparse.Namespace,
    devData: Optional[Data] = None,
    best_score: float = 0) -> None:
    data_sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    dataloader = DataLoader(dataset=dataset, sampler=data_sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)

    total_steps = len(dataloader) // args.grad_accum_steps * args.n_epochs
    if args.local_rank in [-1, 0]:
        logger.info("total steps is %d." % (total_steps))

    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    model.zero_grad()
    global_steps = 0
    best_score = best_score
    for epoch in range(args.n_epochs):
        model.train()
        if args.local_rank in [-1, 0]:
            logger.info("%d / %d EPOCH" % (epoch + 1, args.n_epochs))
        training_loss = 0.0
        logging_loss = 0.0
        for step, batch in enumerate(tqdm(dataloader, desc="%d / %d epoch" % (epoch + 1, args.n_epochs))):
            batch: List[torch.Tensor] = [t.to(args.device) for t in batch]
            memory, dialog_memory, kb_len, dialog_len, resp_len, global_memory_label, sketch_resp, local_memory_label = batch
            loss = model(memory, \
                                dialog_memory, \
                                kb_len, \
                                dialog_len, \
                                resp_len, \
                                sketch_resp, \
                                global_memory_label, \
                                local_memory_label)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.grad_accum_steps > 1:
                loss = loss / args.grad_accum_steps
            training_loss += loss.item()
            loss.backward()
            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
                global_steps += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and (step + 1) % args.logging_steps == 0:
                    log_step = (step + 1) // args.logging_steps
                    log_lr: float = scheduler.get_last_lr()[0]
                    # log_lr: float = args.learning_rate
                    logger.info(
                        "%d / %d epoch - %d / %d step, lr:%f, loss:%f"
                        % (epoch + 1,
                            args.n_epochs,
                            global_steps,
                            total_steps,
                            log_lr,
                            (training_loss - logging_loss) / args.logging_steps)
                    )
                    logging_loss = training_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and (step + 1) % args.save_steps == 0:
                    model1 = model.module if hasattr(model, "module") else model
                    model_name = "%s_%d_epoch_%d_step" % (args.model_name, epoch + 1, log_step)
                    model_path = model1.save_model(model_name)
                    logger.info("%d epoch %d step checkpoint saved to %s." % (epoch + 1, log_step))

        if args.save_checkpoints:
            if args.local_rank in [-1, 0]:
                model1 = model.module if hasattr(model, "module") else model
                modelName = "%s_%d_ckpt" % (args.model_name, epoch + 1)
                model_path = model1.save_model(modelName)
                logger.info("model checkpoint saved to %s." % (model_path))

        if args.do_eval and devData is not None:
            if args.local_rank in [-1, 0]:
                bleu, p, r, f1, sche_p, sche_r, sche_f1, wea_p, wea_r, wea_f1, nav_p, nav_r, nav_f1 = eval(model, dev_data, args)
                logger.info("evaluate %d epoch - bleu: %f, p:%f, r:%f, f1:%f, \
                            sche_p: %f, sche_r: %f, sche_f1: %f, \
                            wea_p: %f, wea_r: %f, wea_f1: %f, \
                            nav_p: %f, nav_r: %f, nav_f1: %f" % (epoch, bleu, p, r, f1, \
                                                                sche_p, sche_r, sche_f1, \
                                                                wea_p, wea_r, wea_f1, \
                                                                nav_p, nav_r, nav_f1))
                score = bleu + f1
                if score > best_score:
                    model1 = model.module if hasattr(model, "module") else model
                    model_name = "%s_best" % (args.model_name)
                    model_path = model1.save_model(model_name)
                    best_score = score
                    logger.info("best model saved to %s." % (model_path))


def eval(model: GLMP, dev_data: Data, args: argparse.Namespace):
    dialogs = dev_data.dialogs
    model.eval()
    with tqdm(total=len(dialogs)) as pbar:
        pbar.set_description("eval")
        total_bleu = 0
        total_p, total_r, total_f1, count = 0, 0, 0, 0 
        sche_p, sche_r, sche_f1, sche_count = 0, 0, 0, 0
        wea_p, wea_r, wea_f1, wea_count = 0, 0, 0, 0
        nav_p, nav_r, nav_f1, nav_count = 0, 0, 0, 0

        for dialog in dialogs:
            pbar.update(1)
            domain = dialog.domain
            utters = dialog.utters
            memory = dialog.knowledge.kb_memory
            history: List[Tuple[str, str]] = []
            for i in range(0, len(utters), 2):
                user_utter = utters[i]
                sys_utter = utters[i+1]
                assert user_utter.role == "user"  
                assert sys_utter.role == "sys"
                history.append(("user", user_utter.sent))
                gold_resp = sys_utter.sent
                gold_resp = gold_resp.split(" ")
                gold_kb = sys_utter.kb_mention
                with torch.no_grad():
                    _, fine_resp = model.predict(memory, history)
                history.append(("sys", sys_utter.sent))
                p, r, f1, c = compute_entity_PRF(fine_resp, gold_kb, args.global_entities)
                bleu = nltk_multi_bleu([gold_resp], fine_resp)

                total_bleu += bleu                
                total_p += p
                total_r += r
                total_f1 += f1
                count += c
                if domain == "schedule":
                    sche_p += p
                    sche_r += r
                    sche_f1 += f1
                    sche_count += c
                elif domain == "navigate":
                    nav_p += p
                    nav_r += r
                    nav_f1 += f1
                    nav_count += c
                elif domain == "weather":
                    wea_p += p
                    wea_r += r
                    wea_f1 += f1
                    wea_count += c
        bleu = total_bleu / count
        p = total_p / count
        r = total_r / count
        f1 = total_f1 / count
        sche_p = sche_p / sche_count
        sche_r = sche_r / sche_count
        sche_f1 = sche_f1 / sche_count
        wea_p = wea_p / wea_count
        wea_r = wea_r / wea_count
        wea_f1 = wea_f1 / wea_count
        nav_p = nav_p / nav_count
        nav_r = nav_r / nav_count
        nav_f1 = nav_f1 / nav_count 

    model.train()
    return bleu, p, r, f1, sche_p, sche_r, sche_f1, wea_p, wea_r, wea_f1, nav_p, nav_r, nav_f1


if __name__ == "__main__":
    args = add_parser()

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = "cuda"
    set_seed(args.seed)

    global_entities = get_global_entities()
    args.global_entities = global_entities

    train_data = Data.load_from_kvr_file(args.train_file)
    dev_data = Data.load_from_kvr_file(args.dev_file)

    best_score = 0
    if args.init_model_dir:
        model = GLMP.create_model(args.init_model_dir, args.device)
        lang = model.lang
        bleu, p, r, f1,sche_p, sche_r, sche_f1, wea_p, wea_r, wea_f1, nav_p, nav_r, nav_f1 = eval(model, dev_data, args)
        logger.info("evaluate init model - bleu: %f, p:%f, r:%f, f1:%f, \
            sche_p: %f, sche_r: %f, sche_f1: %f, \
            wea_p: %f, wea_r: %f, wea_f1: %f, \
            nav_p: %f, nav_r: %f, nav_f1: %f" \
            % (bleu, p, r, f1, \
                sche_p, sche_r, sche_f1, \
                wea_p, wea_r, wea_f1, \
                nav_p, nav_r, nav_f1))
        best_score = bleu + f1
    else:
        lang = Lang.load_from_KVR([args.train_file, args.dev_file])
        model = GLMP(lang, args.hidden, args.memory_hop, args.dropout, args.sent_len)
        model.train()
        model.to(args.device)

    if args.do_train: 
        train_dataset = GLMP_Dataset(lang, train_data, args.memory_len, args.memory_size, args.sent_len)
        if args.local_rank != -1: torch.distributed.barrier()
        train(model, train_dataset, args, dev_data, best_score)