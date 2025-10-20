import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
import copy
import time
from datetime import datetime
from torch.utils.data import Subset, Dataset, DataLoader
import MyDataset
import TestModel
import CreateRerunScript


def create_a_chart(losses, chart_path, title, start_epoch=1):
    plt.figure(figsize=(8, 5))
    epochs = list(range(start_epoch, len(losses) + start_epoch))
    plt.plot(epochs, losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(chart_path, dpi=150)
    plt.close()


def train(model, model_name, part_of_data = 1, start_epoch = 0, best_test_loss = float('inf'), rerun = False):

    start_time = time.time()
    epochs_multiply = 1
    batches = 1


    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"training_{timestamp}"
    model_info_dir = model_name + '//' + log_dir

    os.makedirs(model_info_dir, exist_ok=True)

    log_file = model_info_dir + "//training_log.txt"

    def Log(log_line):
        print(log_line)
        with open(log_file, 'a') as f:
            f.write(log_line + "\n")

    def read_batch_size():
        if os.path.isfile("batch_size.txt"):
            try:
                with open("batch_size.txt", "r") as f:
                    return int(f.read().strip())
            except Exception as e:
                Log(f"Cannot read batch_size.txt: {e}")
                return batches

    if not rerun:
        x = torch.randn(1, 3, 108, 192)
        y = model(x)
        print(y.shape)

    train_dataset = MyDataset.GetTrainDataset()
    test_dataset = MyDataset.GetTestDataset()
    final_test_dataset = MyDataset.GetFinalTestDataset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    best_model_state = model.state_dict()
    best_optimizer_state = optimizer.state_dict()

    num_epochs = 400
    final_epoch = num_epochs

    data_count = int(len(train_dataset) / num_epochs)

    num_epochs = int(num_epochs * part_of_data)

    batches = read_batch_size()
    test_dataloader = DataLoader(test_dataset, batch_size=batches, shuffle=False, drop_last=True)
    final_test_dataloader = DataLoader(final_test_dataset, batch_size=batches, shuffle=False, drop_last=True)

    print("Starting...")
    bad_loss_count = 0
    returning_weight_count = 0

    if rerun:
        Log(f"Reruning training!")
        Log(f"Starting test loss: {best_test_loss:.15f}")

    losses_train = []
    losses_test = []

    lr_multiply = 1

    for epoch in range(start_epoch, num_epochs * epochs_multiply):
        if os.path.isfile("STOP"):
            Log("Training stoped!")
            final_epoch = epoch
            model.load_state_dict(best_model_state)
            optimizer.load_state_dict(best_optimizer_state)
            shutil.move("STOP", "Tools")
            break
        if os.path.isfile("TEST"):
            TestModel.Upscale_83(model, model_info_dir)
            shutil.move("TEST", "Tools")

        new_lr = 0.001

        if os.path.isfile("LearningRate.txt"):
            try:
                with open("LearningRate.txt", "r") as f:
                    new_lr = float(f.read().strip()) * lr_multiply
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            except Exception as e:
                Log(f"Cannot read LearningRate.txt: {e}")

        batches = read_batch_size()

        model.train()
        running_loss = 0.0

        data_start = data_count * (epoch % int(num_epochs * 2.5))
        subset1 = Subset(train_dataset, list(range(data_start, data_start + data_count)))

        dataloader = DataLoader(subset1, batch_size=batches, shuffle=True, drop_last=True)

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        losses_train.append(epoch_loss)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss /= len(test_dataset)
        losses_test.append(test_loss)

        duration = time.time() - start_time

        log_line = f"Epoch [{epoch + 1}/{num_epochs * epochs_multiply}] Lr: {new_lr:.7f} B: {batches} T:{duration:.2f}s Train Loss: {epoch_loss:.15f} Test Loss: {test_loss:.15f}"

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            bad_loss_count = 0
            returning_weight_count = 0
        else:
            bad_loss_count += 1
            log_line += f" => worse loss ({bad_loss_count} attempt)"

        Log(log_line)

        if bad_loss_count > (30 - returning_weight_count * 10) or test_loss > 999:
            model.load_state_dict(best_model_state)
            optimizer.load_state_dict(best_optimizer_state)
            returning_weight_count += 1
            bad_loss_count = 0
            lr_multiply = lr_multiply / 4
            Log(f"Return to best weights ({returning_weight_count}/3)")
            if returning_weight_count >= 3:
                Log(f"Stoping training! Lack of progress in learning.")
                final_epoch = epoch + 1
                break

    model.eval()
    final_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in final_test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            final_test_loss += loss.item() * inputs.size(0)

    final_test_loss /= len(final_test_dataset)

    duration = time.time() - start_time
    Log(f"Duration: {duration} s")

    Log(f"=========================================================================")
    Log(f"Final epoch: {final_epoch}")
    Log(f"Test loss: {best_test_loss:.15f}")
    Log(f"Final test loss: {final_test_loss:.15f}")

    torch.save(model, model_info_dir + '//Model.pth')

    create_a_chart(losses_train, model_info_dir + '//LossesTrain.png', "Losses train", 1)
    create_a_chart(losses_test, model_info_dir + '//LossesTest.png', "Losses test", 1)

    create_a_chart(losses_train[int(len(losses_train)/2):], model_info_dir + '//LossesTrain2.png', "Losses train", int(len(losses_train)/2+1))
    create_a_chart(losses_test[int(len(losses_test)/2):], model_info_dir + '//LossesTest2.png', "Losses test", int(len(losses_test)/2+1))

    shutil.copy2(model_name + '//' + model_name + '.py', model_info_dir + '//' + model_name + '_copy.py')

    scripted = torch.jit.script(model)
    scripted.save(model_info_dir + "//Model.pt")

    CreateRerunScript.Create(model_info_dir + '//Model.pth', best_test_loss, final_epoch, model_info_dir + '//RerunTraining.py')

    TestModel.SimpleTest(model, model_info_dir)

    info_file = model_name + "//BestModel.txt"

    if os.path.exists(info_file):
        with open(info_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            with open(info_file, "w", encoding="utf-8") as f:
                f.write(f"{final_test_loss:.15f} - {log_dir}")
        else:
            first_line = lines[0].strip()
            first_value_str = first_line.split()[0]
            first_value = float(first_value_str)

            if first_value > final_test_loss:
                new_line = f"{final_test_loss:.15f} - {log_dir}\n"

                with open(info_file, "w", encoding="utf-8") as f:
                    f.write(new_line)
                    f.writelines(lines)

                print(f"New best model!")
    else:
        with open(info_file, "w", encoding="utf-8") as f:
            f.write(f"{final_test_loss:.15f} - {log_dir}")
        print(f"New best model!")



