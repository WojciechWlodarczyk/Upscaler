
import ManageTrainings

ModelName = "Model_3/training_2025-08-21_09-08-10/first_Model.pth"
best_test_loss = 0.000057842905877
final_epoch = 134

ManageTrainings.RerunTraining(ModelName, best_test_loss, final_epoch)
