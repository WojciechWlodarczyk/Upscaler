

def Create(modelName, best_test_loss, final_epoch, filename="nowy_skrypt.py"):
    code = f"""
import ManageTrainings

ModelName = "{modelName}"
best_test_loss = {best_test_loss:.15f}
final_epoch = {final_epoch}

ManageTrainings.RerunTraining(ModelName, best_test_loss, final_epoch)
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)