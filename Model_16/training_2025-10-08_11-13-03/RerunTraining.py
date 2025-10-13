
import ManageTrainings

ModelName = "Model_16//training_2025-10-08_11-13-03//Model.pth"
# best_test_loss = 0.000000020035603
# final_epoch = 400

best_test_loss = 999.9
final_epoch = 0

ManageTrainings.RerunTraining(ModelName, best_test_loss, final_epoch)
    