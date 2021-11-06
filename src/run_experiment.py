import plotly.express as px
import torch
from transformers import AutoTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import itertools
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix


def setup(args):
    """do basic setup - load model and tokenizer and setup gpu.

    Args:
        args ([type]): argparse args

    Returns:
        Tuple: tokenizer, Transformer model, device
    """
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-large",
        num_labels=args.num_labels,
        output_attentions=True,
        output_hidden_states=True,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    return tokenizer, model, device


def read_dataset(args):
    """read training dataset

    Args:
        args ([type]): argparse args

    Returns:
        Tuple: list of enrichments, list of sentences, list of labels
    """
    df = pd.read_csv(args.input_csv, sep="\t", engine="python",)
    labs = args.labels.split(",")
    stance_mapping = {x: i for i, x in zip(itertools.count(0), labs)}
    df["Stance"] = df[args.label_clm].replace(stance_mapping)

    enrichments = list(df[args.enrichment_clm].values)
    sentences = list(df[args.label_clm].values)
    labels = torch.tensor(df["Stance"].values)

    print("Targets and Sentences Lengths:", len(enrichments), len(sentences))
    print("sentences[0]:", " ".join(sentences[0].split()))
    print(df.sample(n=10))

    return enrichments, sentences, labels


def generate_train_dataset(args, tokenizer, enrichments, sentences, labels):
    """generate training dataset/dataloader

    Args:
        args ([type]): argparse args
        tokenizer ([type]): HuggingFace tokenizer
        enrichments ([type]): list of enrichments (e.g. targets, entity descriptions)
        sentences ([type]): list of sentences/tweets
        labels ([type]): list of true labels

    Returns:
        Tuple: train dataloader, validation dataloader
    """
    encoded_dict = tokenizer(
        enrichments,
        sentences,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=args.max_tokens,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoded_dict["input_ids"]
    attention_masks = encoded_dict["attention_mask"]

    print(" ".join(sentences[0].split()))
    print(input_ids[0], attention_masks[0], labels[0], sep="\n")

    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Calculate the number of samples to include in each set.
    train_size = int(args.train_set_pct * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("{:>5,} training samples".format(train_size))
    print("{:>5,} validation samples".format(val_size))

    batch_size = args.batch_size

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size,
    )

    validation_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size,
    )

    return train_dataloader, validation_dataloader


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(args, model, device, train_dataloader, validation_dataloader):
    """main training loop

    Args:
        args ([type]): argparse args
        model ([type]): HuggingFace model
        device ([type]): device (cpu or gpu)
        train_dataloader ([type]): training dataloader
        validation_dataloader ([type]): validation dataloader

    Returns:
        [type]: training statistics
    """

    model.cuda()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8,)

    # Number of training epochs. The BERT authors recommend between 2 and 4.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default value in run_glue.py
        num_training_steps=total_steps,
    )

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    seed_val = 43
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")

        t0 = time.time()
        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                        step, len(train_dataloader), elapsed
                    )
                )

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(
                b_input_ids,
                return_dict=True,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    return_dict=True,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                loss = outputs.loss
                logits = outputs.logits

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

    print("")
    print("Training complete!")

    print(
        "Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0))
    )

    return training_stats


def display_stats(training_stats):
    """display the training stats

    Args:
        training_stats ([type]): dictionary of training stats
    """
    pd.set_option("precision", 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index("epoch")
    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.plot(df_stats["Training Loss"], "b-o", label="Training")
    plt.plot(df_stats["Valid. Loss"], "g-o", label="Validation")
    plt.plot(df_stats["Valid. Accur."], "r-o", label="Valid. Accur.")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])
    plt.show()


def save_model(model, tokenizer):
    """save models to disk

    Args:
        model ([type]): HuggingFace model
        tokenizer ([type]): HuggingFace tokenizer
    """
    if not os.path.exists(args.model_output_dir):
        os.makedirs(args.model_output_dir)

    model.save_pretrained(args.model_output_dir)
    tokenizer.save_pretrained(args.model_output_dir)


def load_test_data(args, tokenizer):
    """load test data

    Args:
        args ([type]): argparse args
        tokenizer ([type]): HuggingFace tokenizer

    Returns:
        [type]: test dataloader, true_labels, enrichments, sentences
    """
    df_test = pd.read_csv(args.test_csv, sep="\t")
    labs = args.labels.split(",")
    stance_mapping = {}
    df_test["Stance"] = df_test[args.label_clm].replace(stance_mapping)
    print(df_test.groupby("Stance").count())

    # Encode the labels
    enrichments = list(df_test[args.enrichment_clm].values)
    sentences = list(df_test[args.text_clm].values)
    true_labs = list(df_test["Stance"].values)

    encoded_dict = tokenizer(
        enrichments,
        sentences,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=256,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoded_dict["input_ids"]
    attention_masks = encoded_dict["attention_mask"]

    print("Original: ", sentences[0])
    print("Token IDs:", input_ids[0])
    print(input_ids.shape)
    print(attention_masks.shape)
    print("Tokenized {:,} test sentences...".format(len(input_ids)))

    batch_size = args.batch_size
    test_data = TensorDataset(input_ids, attention_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return test_dataloader, true_labs, enrichments, sentences


def test(test_dataloader):
    """evaluate model on test set

    Args:
        test_dataloader ([type]): test dataloader

    Returns:
        [type]: [description]
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_output_dir)
    model.to(device)
    model.eval()

    cls_tokens_list = []
    predictions = []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        cls_tokens = list(outputs.hidden_states)[-1][:, 0, :]
        cls_tokens_list.append(cls_tokens.cpu())

        logits = outputs.logits.detach().cpu().numpy()
        predictions.append(logits)

    print("    DONE.")

    # get list of class predictions
    pred_labs = np.concatenate(predictions, axis=0)
    print(pred_labs.shape)
    print(pred_labs[:10, :])

    # convert probs to predicted labels
    pred_labs = np.argmax(pred_labs, axis=1).flatten()
    print(pred_labs.shape)
    print(pred_labs)

    return pred_labs


def run_evaluation_metrics(args, enrichments, pred_labs, true_labs, sentences):
    """run evaluation metrics on preidctions

    Args:
        args ([type]): argparse args
        enrichments ([type]): list of enrichments (e.g. targets or entity descriptions)
        pred_labs ([type]): predicted labels
        true_labs ([type]): true labels
        sentences ([type]): sentences/tweets
    """
    sentences_pretty = [" ".join(x.split()) for x in sentences]
    labs = args.labels.split(",")
    print("Target       Predicted  True       Sentence")
    print("------------------------------------------")

    for (enrichment, pred_lab, true_lab, sentence, _) in zip(
        enrichments, pred_labs, true_labs, sentences_pretty, range(50)
    ):
        print(f"{enrichment:12.12} {labs[pred_lab]:10} {labs[true_lab]:10} {sentence}")

    print("Accuracy:", accuracy_score(true_labs, pred_labs))
    print("Micro F1:", f1_score(true_labs, pred_labs, average="micro"))
    print("Macro F1:", f1_score(true_labs, pred_labs, average="macro"))
    print("Matthews:", matthews_corrcoef(true_labs, pred_labs))

    stance_mapping_inv = {i: x for i, x in zip(itertools.count(0), labs)}
    labels = list(sorted(labs))

    true_labs_text = list(map(lambda x: stance_mapping_inv[x], true_labs))
    pred_labs_text = list(map(lambda x: stance_mapping_inv[x], pred_labs))

    print(
        "Precision:",
        precision_score(true_labs_text, pred_labs_text, labels=labels, average=None),
    )
    print(
        "Recall:",
        recall_score(true_labs_text, pred_labs_text, labels=labels, average=None),
    )
    print(
        "Micro F1:",
        f1_score(true_labs_text, pred_labs_text, labels=labels, average=None),
    )

    f_favor = f1_score(true_labs, pred_labs, labels=[0], average=None)
    f_against = f1_score(true_labs, pred_labs, labels=[1], average=None)
    score = (f_favor + f_against) / 2.0
    print("SemEval-Score:", score)

    conf_mat = confusion_matrix(true_labs, pred_labs)
    conf_mat_norm = [x / sum(x) for x in conf_mat]

    print(conf_mat)

    px.imshow(
        conf_mat_norm,
        labels=dict(x="Predicted Label", y="True Label"),
        x=["Favour", "Against", "Neither", "Neutral", "Mix"],
        y=["Favour", "Against", "Neither", "Neutral", "Mix"],
        color_continuous_scale="greens",
        range_color=(0, 1),
        title="Confusion Matrix",
    )


def run(args):
    """main function

    Args:
        args ([type]): argparse args
    """
    tokenizer, model, device = setup(args)
    targets, sentences, labels = read_dataset(args)
    train_dl, val_dl = generate_train_dataset(
        args, tokenizer, targets, sentences, labels
    )
    stats = train(args, model, device, train_dl, val_dl)
    display_stats(stats)
    save_model(model, tokenizer)
    test_dl, true_labs, targets, sentences = load_test_data(args, tokenizer)
    pred_labs = test(device, test_dl)
    run_evaluation_metrics(args, targets, pred_labs, true_labs, sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_csv",
        default="/content/drive/My Drive/data/grimminger/train-enriched-stancy.tsv",
    )
    parser.add_argument(
        "test_csv",
        default="/content/drive/My Drive/data/grimminger/test-enriched-stancy.tsv",
    )
    parser.add_argument(
        "text_clm", default="text",
    )
    parser.add_argument(
        "label_clm", default="Trump",
    )
    parser.add_argument(
        "enrichment_clm", default="Description",
    )
    parser.add_argument(
        "batch_size", type=int, default=8,
    )
    parser.add_argument(
        "max_tokens", type=int, default=256,
    )
    parser.add_argument(
        "train_set_pct", type=float, default=0.9,
    )
    parser.add_argument(
        "num_labels", type=int, default=5,
    )
    parser.add_argument("labels", default="Favor,Against,Neither")
    parser.add_argument(
        "model_output_dir", default="",
    )
    args = parser.parse_args()
    run(args)
