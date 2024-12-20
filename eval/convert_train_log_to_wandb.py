"""
Accepts the training log JSON obtained post BitDistiller training. It will
automatically parse the file and upload the loss statistics to the provided wandb
account.
"""

import argparse
import wandb
import json

"""
Helper function to convert the JSON path into
a python object
"""
def get_json_object(json_path):
    json_object = None
    with open(json_path) as file:
        json_object = json.load(file)
    return json_object

def upload_to_wandb(train_json):
    # Get general training info
    log_data = train_json["log_history"]
    epochs = train_json["epoch"]
    lr = log_data[0]["learning_rate"]

    # Init Weights and Biases
    wandb.init(
        project="hpml-training-logs",
        config={
            "learning_rate": lr,
            "epochs": epochs
        }
    )

    # The log file mixes eval_loss and train_loss together, but we can separate them
    # by whether the eval_loss key shows up in the dict for the log step
    for log in log_data:
        if "eval_loss" in log:
            wandb.log({"eval_loss": log["eval_loss"]})
        elif "loss" in log:
            wandb.log({"train_loss": log["loss"]})

    wandb.finish()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="log to wandb parser")
    parser.add_argument("--log_path", help="path to training log")

    args = parser.parse_args()

    train_json = get_json_object(args.log_path)
    upload_to_wandb(train_json)