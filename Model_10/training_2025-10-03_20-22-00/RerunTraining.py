
import ManageTrainings

ModelName = "Model_10//training_2025-10-03_20-22-00//Model.pth"
best_test_loss = 0.005363315090071
final_epoch = 400

ManageTrainings.RerunTraining(ModelName, best_test_loss, final_epoch)
    